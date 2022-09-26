moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]

delta(i) = maximum([0, i]) > 0 ? 1 : 0

function getres(P)
    global Pm = P
    result = optimize(sqerror, x0, LBFGS())
    min_val = Optim.minimum(result)
    return Pm, result, min_val
end

function ∇f(G::AbstractVector{T}, X::T...) where {T}

    # Exact analytical solutions to derivatives
    @Symbolics.variables E P_0 I_inf P_b Rcsf
    f = P_0 + (((P_b - P_0) / Rcsf) + I_inf) * (P_b + P_0) / (((P_b - P_0) / Rcsf) + I_inf*exp(-E*(((P_b - P_0) / Rcsf) + I_inf)))

    dRcsf = Differential(Rcsf)
    df = expand_derivatives(dRcsf(f))
    fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2]))
    G[1] = Symbolics.value(fval)

    dE = Differential(E)
    df = expand_derivatives(dE(f))
    fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2]))
    G[2] = Symbolics.value(fval)

    dP0 = Differential(P_0)
    df = expand_derivatives(dP0(f))
    fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2]))
    G[3] = Symbolics.value(fval)

    # P_b = Data["P_b"]
    # I_inf = Data["I_inf"]
    # Rcsf = X[1]
    # E = X[2]
    # P_0 = X[3]

    # G[1] = -(P_0 + P_b) * (-P_0 + P_b) / (Rcsf^2 * ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf)) - (P_0 + P_b) * (I_inf + (-P_0 + P_b) / Rcsf) * (-(-P_0 + P_b) / Rcsf^2 + E * exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * (-P_0 + P_b) * I_inf / Rcsf^2) / ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf)^2
    # G[2] = exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * (P_0 + P_b) * I_inf * (I_inf + (-P_0 + P_b) / Rcsf)^2 / ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf)^2
    # G[3] = 1 + (I_inf + (-P_0 + P_b) / Rcsf) / ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf) - (P_0 + P_b) / (Rcsf * ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf)) - (P_0 + P_b) * (I_inf + (-P_0 + P_b) / Rcsf) * (E * exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf / Rcsf - Rcsf^(-1)) / ((-P_0 + P_b) / Rcsf + exp(-E * (I_inf + (-P_0 + P_b) / Rcsf)) * I_inf)^2

    return
end

function exceldatetodate(exceldate::Real)
    t, d = modf(exceldate)
    return Dates.DateTime(1899, 12, 30) + Dates.Day(d) + Dates.Millisecond((floor(t * 86400000)))
end

function parseXML(Parameters, Selections, Results)

    selections = Dict()
    results = Dict()

    for i in 1:length(Selections)
        sel = elements(Selections[i])
        sel_name = Selections[i]["Name"]
        sel_st = sel[1]["StartTime"]
        sel_en = sel[1]["EndTime"]

        sel_st = DateTime(sel_st, dateformat"dd/mm/yyyy HH:MM:SS")
        sel_en = DateTime(sel_en, dateformat"dd/mm/yyyy HH:MM:SS")

        selections = merge!(selections, Dict(sel_name => [sel_st, sel_en]))
    end

    for i in 1:length(Results)
        res = elements(Results[i])
        res_name = Results[i]["Name"]
        name_surrogate = split(res_name, " ")

        # Some parameters name have units e.g. [mmHg] but Dict field naming does not support strings with non-alphanumeric chars
        if isletter(name_surrogate[2][1])
            res_name = name_surrogate[1] * "_" * name_surrogate[2]
        else
            res_name = name_surrogate[1]
        end

        res_val = Results[i]["Value"]
        results = merge!(results, Dict(res_name => parse(Float64, res_val)))
    end
    return results, selections
end

function readCSF(filename)

    recording_start_time = 0.0
    recording_end_time = 0.0

    fid = h5open(filename, "r")

    # Get attributes outside of datasets/groups
    completeAttrFlg = false
    try # For some reason some data exported to hdf5 does not have these attributes so need to calculate ad hoc
        recording_end_time = read_attribute(fid, "dataEndTime")
        recording_start_time = read_attribute(fid, "dataStartTime")
        dur = read_attribute(fid, "duration")
        dur = split(dur[1], " ")
        completeAttrFlg = true
    catch # Obtain these from ICP datetime & parse later on when it is loaded
        recording_start_time = [DateTime(2013)]
        recording_end_time = [DateTime(2013)]   # Pre-allocate so the type works
    end

    # Begin handling data in XML string - some files do not have infusion test output
    xml_obj = fid["aux/ICM+/icmtests"]
    xml_data = read(xml_obj)
    xml_string = String(xml_data[1])
    data = parsexml(xml_string)
    icm_data = root(data)
    vars = elements(icm_data)

    SingleAnalysis = firstelement(vars[2])
    SingleAnalysis = elements(SingleAnalysis)
    # Variables = elements(SingleAnalysis[1]) # Not currently used
    Selections = elements(SingleAnalysis[2])
    Parameters = elements(SingleAnalysis[3])
    Results = elements(SingleAnalysis[4])

    results, selections = parseXML(Parameters, Selections, Results)

    # This dereferencing is pain, there has to be a more elegant solution
    t_series_ds = fid["summaries/minutes/"]
    t_series = t_series_ds[:]
    numsamples = length(t_series)

    # Pre-allocate ?needed
    ICP = zeros(numsamples)
    AMP = zeros(numsamples)
    timestamp = zeros(numsamples)
    P0 = zeros(numsamples)
    AMP_P = zeros(numsamples)

    # Dereferencing named tuple...
    for i in 1:numsamples
        ICP[i] = t_series[i].ICP
        AMP[i] = t_series[i].AMP
        timestamp[i] = t_series[i].datetime
        # Some files do not have P_0 saved?
        try
            P0[i] = t_series[i].P0
            AMP_P[i] = t_series[i].AMP_P
        catch
        end
    end

    if !completeAttrFlg # If for some reason the recording start and end times are not saved, obtain them from timestamp data
        recording_start_time[1] = exceldatetodate(timestamp[1])
        recording_end_time[1] = exceldatetodate(timestamp[end])
        start_time = recording_start_time[1]
        end_time = recording_end_time[1]
    else
        start_time = DateTime(recording_start_time[1], dateformat"yyyy/mm/dd HH:MM:SS")
        end_time = DateTime(recording_end_time[1], dateformat"yyyy/mm/dd HH:MM:SS")
    end

    # Because not all files have E saved
    try
        global E = results["Elasticity"]
    catch
        global E = 0.11
    end

    Data = Dict{String,Any}()
    # Data["P_0"] = results["Pss"]
    Data["I_b"] = results["CSF_production"]
    Data["Rcsf"] = results["Rcsf"]

    try
        Data["P_0"] = results["Pss"]
    catch
        Data["P_0"] = results["pss"]
    end

    Data["E"] = E
    Data["ICP"] = ICP
    Data["AMP"] = AMP
    Data["P_p"] = results["ICP_plateau"]
    Data["P_b"] = results["ICP_baseline"]
    Data["T"] = [0:numsamples-1...] * 1 / 6
    Data["infusion_start_frame"] = round(Int, (selections["Infusion"][1] - start_time).value / 10000)
    Data["infusion_end_frame"] = round(Int, (selections["Infusion"][2] - start_time).value / 10000)
    Data["plateau_start"] = round(Int, (selections["Plateau"][1] - start_time).value / 10000)
    Data["plateau_end"] = round(Int, (selections["Plateau"][2] - start_time).value / 10000)
    Data["rec_dur_s"] = (end_time - start_time).value / 1000
    Data["start_time"] = start_time
    Data["end_time"] = end_time
    Data["I_inf"] = parse(Float64, Parameters[1]["Value"])
    Parameters[3]["Value"] == "yes" ? Data["one_needle"] = 1 : Data["one_needle"] = 0
    Data["one_needle"] == 1 ? Data["Rn"] = results["Needle_resist."] : Data["Rn"] = 0
    return Data
end

function plot_model(I_b, E, P_0, ICP, dsampf, trend)
    infusion_start_frame = Data["infusion_start_frame"]
    infusion_end_frame = Data["infusion_end_frame"]
    plateau_start = Data["plateau_start"]
    plateau_end = Data["plateau_end"]
    P_b = Data["P_b"]
    P_p = Data["P_p"]
    I_inf = Data["I_inf"]
    numsamples = length(ICP)
    infusion_end_frame > numsamples ? (global infusion_end_frame = numsamples) : 0
    # println("Estimated parameters:\nIₐ = $Ibr [mL/min]\n" * "E = $Er [mmHg/mL]\n" * "P₀ = $P0r [mmHg]\n")

    gg = moving_average(ICP, dsampf)
    g0 = zeros(length(ICP))
    g0 .+= P_b
    g0[Int(dsampf / 2):Int(dsampf / 2)+length(gg)-1] = gg
    g0[Int(dsampf / 2)+length(gg):end] .= P_p

    P_m = zeros(numsamples)
    P_m .+= P_b
    P_model = zeros(numsamples)
    ICPm = zeros(infusion_end_frame - infusion_start_frame)

    errorVal = 0.0
    for i = infusion_start_frame:infusion_end_frame
        tᵢ = (i - infusion_start_frame) / 6
        It = I_b + I_inf
        ΔP = P_b - P_0
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * tᵢ))) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (ICP[i] - y)^2
        P_model[i] = y
    end
    global fitErrorVal = 100 * sqrt(errorVal) / length(ICPm) / abs(mean(ICP[infusion_start_frame:infusion_end_frame]))

    ICPm = P_model[infusion_start_frame:infusion_end_frame]
    P_m[infusion_start_frame:infusion_end_frame] = ICPm
    P_m[infusion_end_frame+1:end] .= ICPm[end]

    # plateau_end=numsamples
    vline([infusion_start_frame], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:white, alpha=0.5, linewidth=1, label="Start of infusion")
    vline!([infusion_end_frame], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:white, alpha=0.5, linewidth=1, label="End of infusion")
    vline!([plateau_start], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:mint, alpha=0.5, linewidth=1, label="Start of plateau")
    hline!([P_p], linecolor=:coral2, label="Pₚ", linewidth=0.5, alpha=0.5)
    trend ? plot!(g0, linewidth=2, alpha=0.8, linecolor=:violet, label="Moving average") : 0 # Plot moving average

    plot!(ICP, linecolor=:cadetblue, linewidth=2, label="Measured", alpha=0.7) # Plot ICP from beginning until end of plateau
    # Plot model prediction from beginning until end of plateau
    plot!(P_m, linecolor=:orange, linewidth=2, linestyle=:dash, xlims=[1, plateau_end], ylims=[minimum(ICP) * 0.9, maximum(ICP) * 1.1], xlabel="Time [min]", ylabel="ICP [mmHg]", xticks=([0:30:plateau_end;], [0:30:plateau_end;] ./ 10),
        label="Model", grid=false, titlefontsize=8, titlealign=:left, background=RGB(0.13, 0.14, 0.14))
    title!("I_b = $I_b\n" * "Rcsf = $(value(Rcsf))\n" * "E = $(value(E))\n" * "P_0 = $(value(P_0)))\n" * "error = $fitErrorVal")
end

function errfun(Rcsf::Real, E::Real, P_0::Real)
    errorVal = 0.0
    ΔP = Data["P_b"] - P_0
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    δlb = delta.(Ib_lower .- I_b)
    δub = delta.(I_b .- Ib_upper)
    δ = C .* vcat(δlb, δub)
    penalty = sum(δ .^ κ)

    # global fitErrorVal = 100 * (sqrt(errorVal) / length(Pm) / abs(mean(Pm)))
    return errorVal + penalty
end

function errfunBayes(x)
    Rcsf = x[1]
    E = x[2]
    P_0 = x[3]

    errorVal = 0.0

    ΔP = Data["P_b"] - P_0
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    δlb = delta.(Ib_lower .- I_b)
    δub = delta.(I_b .- Ib_upper)
    δ = C .* vcat(δlb, δub)
    penalty = sum(δ .^ κ)
    # global fitErrorVal = 100 * (sqrt(errorVal) / length(Pm) / abs(mean(Pm)))
    return errorVal + penalty
end

function getModelNL(lowerbound, upperbound, optalg)
    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", optalg)
    register(model, :errfun, 3, errfun, ∇f)

    @variable(model, lowerbound[1] <= Rcsf <= upperbound[1])
    @variable(model, lowerbound[2] <= E <= upperbound[2])
    @variable(model, lowerbound[3] <= P_0 <= upperbound[3])
    # @NLconstraint(model, c1, (P_b - P_0) / Rcsf <= 0.5)
    @NLobjective(model, Min, errfun(Rcsf, E, P_0))

    set_start_value(Rcsf, Data["Rcsf"])
    set_start_value(E, minimum([Data["E"], 1.0]))
    set_start_value(P_0, minimum([0.0, Data["P_b"]]))

    JuMP.optimize!(model)
    # optimize!(model)
    return value(Rcsf), value(E), value(P_0)
end

function getModelBayes(lowerbound, upperbound, bkernel, bsctype, bltype)

    config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
    # set_kernel!(config, "kMaternARD5")  # calls set_kernel of the C API - more accurate
    set_kernel!(config, bkernel)  # calls set_kernel of the C API
    config.sc_type = bsctype # maximum a posteriori method
    config.noise = 1.0e-9
    # config.l_type = bltype # Markov Chain Monte Carlo
    config.n_inner_iterations = 200
    config.n_init_samples = 100
    config.random_seed = 1
    config.force_jump = 50
    config.verbose_level = 0

    optimizer, optimum = bayes_optimization(errfunBayes, lowerbound, upperbound, config)

    Rcsf = optimizer[1]
    E = optimizer[2]
    P_0 = optimizer[3]

    return Rcsf, E, P_0
end