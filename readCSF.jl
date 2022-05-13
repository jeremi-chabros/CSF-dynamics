moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]

delta(i) = maximum([0, i]) > 0 ? 1 : 0

function getres(P)
    global Pm = P
    result = optimize(sqerror, x0, LBFGS())
    min_val = Optim.minimum(result)
    return Pm, result, min_val
end

function ∇f(G::AbstractVector{T}, X::T...) where {T}
    I_b = X[1]
    E = X[2]
    P_0 = X[3]

    It = I_inf + I_b
    ΔP = P_b - P_0
    Q = I_inf * exp(E * t * It)

    # G[1] = ΔP/(Q+I_b)-(ΔP*It*(1-E*I_inf*t*exp(-E*t*It)))/(Q+I_b)^2
    # G[2] = (It^2*ΔP*t*Q)/(I_b*exp(It*t*E)+I_inf)^2
    # G[3] = 1-It/(Q+I_b)

    G[1] = -(I_inf * (P_b - P_0) * exp(E * (I_b + I_inf)) * (exp(E * (I_b + I_inf)) - E * I_b - E * I_inf - 1)) / (I_b * exp(E * (I_b + I_inf)) + I_inf)^2
    G[2] = (I_inf * (I_inf + I_b)^2 * (P_b - P_0) * exp((I_inf + I_b) * E)) / (I_b * exp((I_inf + I_b) * E) + I_inf)^2
    G[3] = 1 - (I_inf + I_b) / (I_inf * exp(-E * (I_inf + I_b)) + I_b)
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
        P0[i] = t_series[i].P0
        AMP_P[i] = t_series[i].AMP_P
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
    Data["P_0"] = results["Pss"]
    Data["I_b"] = results["CSF_production"]
    Data["E"] = E
    Data["ICP"] = ICP
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

    return Data
end

function plot_model(I_b, E, P_0, ICP, dsampf)

    println("Estimated parameters:\nIₐ = $I_b [mL/min]\n" * "E = $E [mmHg/mL]\n" * "P₀ = $P_0 [mmHg]\n")

    gg = moving_average(ICP, dsampf)
    g0 = zeros(length(ICP))
    g0 .+= P_b
    g0[Int(dsampf / 2):Int(dsampf / 2)+length(gg)-1] = gg
    g0[Int(dsampf / 2)+length(gg):end] .= P_p

    P_m = zeros(numsamples)
    P_m .+= P_b
    P_model = zeros(numsamples)
    ICPm = zeros(infusion_end_frame - infusion_start_frame)

    for i = infusion_start_frame:infusion_end_frame
        tᵢ = (i - infusion_start_frame) / 6
        It = I_b + I_inf
        ΔP = P_b - P_0
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * tᵢ))) + P_0
        P_model[i] = y
    end

    ICPm = P_model[infusion_start_frame:infusion_end_frame]
    P_m[infusion_start_frame:infusion_end_frame] = ICPm
    P_m[infusion_end_frame+1:end] .= ICPm[end]
    # P_m[infusion_end_frame+1:end] .= P_p

    # plateau_end=numsamples
    vline([infusion_start_frame], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:white, alpha=0.5, linewidth=1, label="Start of infusion")
    vline!([infusion_end_frame], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:white, alpha=0.5, linewidth=1, label="End of infusion")
    vline!([plateau_start], background=:transparent, legend=:outertopright, linestyle=:dash, linecolor=:mint, alpha=0.5, linewidth=1, label="Start of plateau")
    hline!([P_p], linecolor=:coral2, label="Pₚ", linewidth=0.5, alpha=0.5)
    plot!(g0, linewidth=2, alpha=0.8, linecolor=:violet, label="Moving average") # Plot moving average
    plot!(ICP, linecolor=:cadetblue, linewidth=2, label="Measured", alpha=0.7) # Plot ICP from beginning until end of plateau
    # Plot model prediction from beginning until end of plateau
    plot!(P_m, linecolor=:orange, linewidth=2, linestyle=:dash, xlims=[1, plateau_end], ylims=[minimum(ICP)*0.9,maximum(ICP)*1.1],
        xlabel="Time [min]", ylabel="ICP [mmHg]", xticks=([0:30:plateau_end;], [0:30:plateau_end;] ./ 10),
        label="Model", grid=false)
end

function getModel(optalg, x0)
    model = Model(NLopt.Optimizer) # Initiate instance of model object
    set_optimizer_attribute(model, "algorithm", optalg) # Set optimization algorithm

    register(model, :myerrfun, 3, myerrfun, ∇f)

    @variable(model, 0.0 <= I_b <= 1.0)
    @variable(model, 0.0 <= E <= 1.0)
    @variable(model, -5.0 <= P_0 <= P_b)

    @NLobjective(model, Min, myerrfun(I_b, E, P_0))

    set_start_value(I_b, x0[1])
    set_start_value(E, x0[2])
    set_start_value(P_0, x0[3])

    JuMP.optimize!(model)

    return model, value(I_b), value(E), value(P_0)
end