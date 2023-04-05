moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]

delta(xnum) = maximum([0, xnum]) > 0 ? 1 : 0

function local_opt(x0, optalg)
    # result = Optim.optimize(ferror, g!, x0, optalg)
    result = Optim.optimize(ferror, x0, optalg)
    min_val = Optim.minimum(result)
    return result, min_val
end

function ferror(X)
    errorVal = 0.0
    Rcsf = X[1]
    E = X[2]
    P_0 = X[3]
    # Rcsf = logit(X[1], lb[1], ub[1])
    # E = logit(X[2], lb[2], ub[2])
    # P_0 = logit(X[3], lb[3], ub[3])

    ΔP = Data["P_b"] - P_0
    # I_b = logit(ΔP / Rcsf, Ib_lower, Ib_upper)
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 6:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    return sqrt(errorVal / length(Pm))
end

function ferrorPss(X)
    errorVal = 0.0
    Rcsf = X[1]
    E = X[2]
    P_0 = X[3]
    Pss = X[4]
    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    return sqrt(errorVal / length(Pm))
end

function ferrorStaticP0(X)
    errorVal = 0.0
    Rcsf = X[1]
    E = X[2]
    Pss = X[3]
    P_0 = Data["P_0"]
    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    return sqrt(errorVal / length(Pm))
end

function ∇f(G::AbstractVector{T}, X::T...) where {T}

    P_b = Data["P_b"]
    I_inf = Data["I_inf"]
    # Rcsf = X[1]
    # E = X[2]
    # P_0 = X[3]
    # Pss = X[4]

    # Exact analytical solutions to derivatives
    Symbolics.@variables E P_0 I_inf P_b Rcsf Pss
    # f = P_0 + (((P_b - Pss) / Rcsf) + I_inf) * (P_b + P_0) / (((P_b - Pss) / Rcsf) + I_inf*exp(-E*(((P_b - Pss) / Rcsf) + I_inf)))
    f = P_0 + (((P_b - P_0) / Rcsf) + I_inf) * (P_b + P_0) / (((P_b - P_0) / Rcsf) + I_inf * exp(-E * (((P_b - P_0) / Rcsf) + I_inf)))

    dRcsf = Differential(Rcsf)
    df = expand_derivatives(dRcsf(f))
    # fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>Data["P_0"], E=>X[2], Pss=>X[3]))
    fval = substitute(df, Dict(I_inf => Data["I_inf"], P_b => Data["P_b"], Rcsf => X[1], P_0 => X[3], E => X[2]))
    G[1] = Symbolics.value(fval)

    dE = Differential(E)
    df = expand_derivatives(dE(f))
    # fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>Data["P_0"], E=>X[2], Pss=>X[3]))
    fval = substitute(df, Dict(I_inf => Data["I_inf"], P_b => Data["P_b"], Rcsf => X[1], P_0 => X[3], E => X[2]))
    G[2] = Symbolics.value(fval)

    dP0 = Differential(P_0)
    df = expand_derivatives(dP0(f))
    # fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2], Pss=>X[4]))
    fval = substitute(df, Dict(I_inf => Data["I_inf"], P_b => Data["P_b"], Rcsf => X[1], P_0 => X[3], E => X[2]))
    G[3] = Symbolics.value(fval)

    # dPss = Differential(Pss)
    # df = expand_derivatives(dPss(f))
    # fval = substitute(df, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>Data["P_0"], E=>X[2], Pss=>X[3]))
    # G[3] = Symbolics.value(fval)

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

    # Find which test is CSF infusion 
    ToolName = elements(parentelement(vars[1]))
    ToolName = elements(ToolName[1])
    for tn = 1:length(ToolName)
        tool = ToolName[tn]["Name"]
        tool == "CSF Infusion Test" ? (global ToolID = tn) : 0
    end

    SingleAnalysis = elements(vars[2])
    SingleAnalysis = SingleAnalysis[ToolID]
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
        return
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

    # TODO: transition period 

    Data["infusion_start_frame"] = round(Int, (selections["Infusion"][1] - start_time).value / 10000)
    Data["infusion_end_frame"] = round(Int, (selections["Infusion"][2] - start_time).value / 10000)
    Data["transition_start_frame"] = round(Int, (selections["Transition"][1] - start_time).value / 10000)
    Data["transition_end_frame"] = round(Int, (selections["Transition"][2] - start_time).value / 10000)
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

function press_vol_curve(Rcsf, P_0, Pm)
    P_b = Data["P_b"]
    I_inf = Data["I_inf"]

    ΔP = P_b - P_0
    I_b = ΔP / Rcsf

    dpress = zeros(length(Pm))
    dvol = zeros(length(Pm))

    for i = 2:length(Pm)
        dvol[i] = dvol[i-1] + (I_inf + I_b - (Pm[i] - P_0) / Rcsf) * 1 / 6
        dpress[i] = (Pm[i] - P_0) / (P_b - P_0)
    end

    volRes = dvol[dpress.>0]
    pressRes = dpress[dpress.>0]

    # Remove volume infused after reaching plateau or the last x% of infusion
    volTotal = Data["infusion_end_frame"] - Data["infusion_start_frame"]
    volLower = Int64(floor(volTotal * 0.1))
    volUpper = Int64(floor(volTotal * 0.5))
    idxrm = (Data["infusion_end_frame"] - Data["plateau_start"]) # in frames

    volRes = volRes[volLower:idxrm]
    pressRes = pressRes[volLower:idxrm]
    # volRes = volRes[volLower:volUpper]
    # pressRes = pressRes[volLower:volUpper]
    
    y = log.(pressRes)
    x = volRes

    coefval = CurveFit.curve_fit(LinearFit, x, y)
    fitted_curve = coefval.(x)

    SSE = sum((fitted_curve.-y).^2)

    dfx = DataFrame(x=x, y=y)
    pv_model = lm(@formula(y ~ x), dfx)
    R2 = r2(pv_model)
    # residuals = GLM.residuals(pv_model)
    # SSE = sum(residuals.^2)

    return volRes, pressRes, fitted_curve, R2, SSE
end

function r_squared(y, fitted_curve)
    ydash = mean(y)
    SSres = sum((y .- fitted_curve) .^ 2)
    SStot = sum((y .- ydash) .^ 2)
    R2 = 1 - (SSres / SStot)
    return R2, SSres
end

function errfun(Rcsf::Real, E::Real, P_0::Real)
    errorVal = 0.0
    penalty = 0.0
    ΔP = Data["P_b"] - P_0
    # I_b = ΔP / (Rcsf * Data["I_inf"])
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        # Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * (1 + I_b) * tᵢ * Data["I_inf"])) + P_0
        errorVal += (Pm[i] - Pᵢ)^2
    end

    # δlb = delta.(Ib_lower .- I_b)
    # δub = delta.(I_b .- Ib_upper)
    # δ = C .* vcat(δlb, δub)
    # penalty = sum(δ .^ κ)
    # I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    # I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    # penalty = δub + δlb
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # penalty+= (errorVal + penalty) / (-log(1-abs(R2))*10000)
    return errorVal + penalty
end

function errfunPss(Rcsf::Real, E::Real, P_0::Real, Pss::Real)
    errorVal = 0.0
    penalty = 0.0
    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    # davson = Rcsf * I_b + P_0 # Look into again
    # davson < Data["P_p"] ? δR = -log(Data["P_p"] - davson) : δR = 10^15


    # I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    # I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    # penalty = δub + δlb + δR
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # penalty+= (errorVal + penalty) / (-log(1-abs(R2))*100)
    return errorVal + penalty
end

function errfunStaticP0(Rcsf::Real, E::Real, Pss::Real)
    errorVal = 0.0
    penalty = 0.0
    P_0 = Data["P_0"]
    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        errorVal += (Pm[i] - Pᵢ)^2
    end
    # davson = Rcsf * I_b + P_0
    # davson < Data["P_p"] ? δR = -log(Data["P_p"] - davson) : δR = 10^15
    I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    penalty = δub + δlb
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # penalty += penalty / (-log(1-abs(R2))*100)
    return errorVal + penalty
end

function errfunBayesPss(x)
    Rcsf = x[1]
    E = x[2]
    P_0 = x[3]
    Pss = x[4]

    errorVal = 0.0
    penalty = 0.0

    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        # errorVal += (Pm[i] - Pᵢ)^2
        errorVal += sqrt((Pm[i] - Pᵢ)^2) # Use either RMSE or this
    end
    # δlb = delta.(Ib_lower .- I_b)
    # δub = delta.(I_b .- Ib_upper)
    # δ = C .* vcat(δlb, δub)
    # penalty = sum(δ .^ κ)

    # davson = Rcsf * I_b + P_0
    # davson < Data["P_p"] ? δR = -log(Data["P_p"] - davson) : δR = 10^15
    I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    penalty = δub + δlb
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # return (errorVal + penalty)/R2
    # return (errorVal + penalty) / (-log(1-abs(R2))*100)
    return errorVal + penalty
end

function errfunBayesStaticP0(x)
    Rcsf = x[1]
    E = x[2]
    P_0 = Data["P_0"]
    Pss = x[3]

    errorVal = 0.0
    penalty = 0.0

    ΔP = Data["P_b"] - Pss
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * (Data["P_b"] - P_0) / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        # errorVal += (Pm[i] - Pᵢ)^2
        errorVal += sqrt((Pm[i] - Pᵢ)^2) # Use either RMSE or this
    end
    # δlb = delta.(Ib_lower .- I_b)
    # δub = delta.(I_b .- Ib_upper)
    # δ = C .* vcat(δlb, δub)
    # penalty = sum(δ .^ κ)

    # davson = Rcsf * I_b + P_0
    # davson < Data["P_p"] ? δR = -log(Data["P_p"] - davson) : δR = 10^15
    I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    penalty = δub + δlb
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # return (errorVal + penalty)/R2
    # return (errorVal + penalty) / (-log(1-abs(R2))*100)
    return errorVal + penalty
end

function errfunBayes(x)
    Rcsf = x[1]
    E = x[2]
    P_0 = x[3]

    errorVal = 0.0
    penalty = 0.0

    ΔP = Data["P_b"] - P_0
    I_b = ΔP / Rcsf
    It = I_b + Data["I_inf"]
    for i = 1:length(Pm)
        tᵢ = (i - 1) / 6
        Pᵢ = It * ΔP / (I_b + Data["I_inf"] * exp(-E * It * tᵢ)) + P_0 + (Data["I_inf"] * Data["Rn"])
        # errorVal += (Pm[i] - Pᵢ)^2 # BayesOpt does not work well with MSE or SSE
        errorVal += sqrt((Pm[i] - Pᵢ)^2) # Use either RMSE or this
    end
    # errorVal = sqrt(errorVal / length(Pm))
    # davson = Rcsf * I_b + P_0
    # davson < Data["P_p"] ? δR = -log(Data["P_p"] - davson) : δR = 10^15
    I_b < Ib_upper ? δub = -log(Ib_upper - I_b) : δub = 10^15
    I_b > Ib_lower ? δlb = -log(I_b - Ib_lower) : δlb = 10^15
    penalty = δub + δlb
    # volRes, pressRes, fitted_curve, R2, MSE = press_vol_curve(Rcsf, P_0)
    # return (errorVal + penalty)/R2
    # return (errorVal + penalty) / (-log(1-abs(R2))*100)
    return errorVal + penalty
end

function getModelNL(lowerbound, upperbound, optalg, x0)
    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", optalg)
    # set_optimizer_attribute(model, "local_optimizer", :LD_LBFGS)
    register(model, :errfun, 3, errfun, ∇f)
    # register(model, :errfun, 3, errfun)

    @variable(model, lowerbound[1] <= Rcsf <= upperbound[1])
    @variable(model, lowerbound[2] <= E <= upperbound[2])
    @variable(model, lowerbound[3] <= P_0 <= upperbound[3])
    # @NLconstraint(model, c1, (Data["P_b"] - P_0) / Rcsf <= 1.0)
    @NLobjective(model, Min, errfun(Rcsf, E, P_0))

    set_start_value(Rcsf, x0[1])
    set_start_value(E, x0[2])
    set_start_value(P_0, x0[3])

    JuMP.optimize!(model)
    return value(Rcsf), value(E), value(P_0)
end

function getModelNLPss(lowerbound, upperbound, optalg, x0)
    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", optalg)
    set_optimizer_attribute(model, "local_optimizer", :LD_LBFGS)
    register(model, :errfunPss, 4, errfunPss, ∇f)

    @variable(model, lowerbound[1] <= Rcsf <= upperbound[1])
    @variable(model, lowerbound[2] <= E <= upperbound[2])
    @variable(model, lowerbound[3] <= P_0 <= upperbound[3])
    @variable(model, lowerbound[4] <= Pss <= upperbound[4])

    @NLobjective(model, Min, errfunPss(Rcsf, E, P_0, Pss))

    set_start_value(Rcsf, x0[1])
    set_start_value(E, x0[2])
    set_start_value(P_0, x0[3])
    set_start_value(Pss, x0[4])

    JuMP.optimize!(model)
    return value(Rcsf), value(E), value(P_0), value(Pss)
end

function getModelStaticP0(lowerbound, upperbound, optalg, x0)
    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", optalg)
    set_optimizer_attribute(model, "local_optimizer", :LD_LBFGS)
    register(model, :errfunStaticP0, 3, errfunStaticP0, ∇f)

    @variable(model, lowerbound[1] <= Rcsf <= upperbound[1])
    @variable(model, lowerbound[2] <= E <= upperbound[2])
    @variable(model, lowerbound[3] <= Pss <= upperbound[3])

    @NLobjective(model, Min, errfunStaticP0(Rcsf, E, Pss))

    set_start_value(Rcsf, x0[1])
    set_start_value(E, x0[2])
    set_start_value(Pss, x0[3])

    JuMP.optimize!(model)
    return value(Rcsf), value(E), value(Pss)
end

function getModelBayes(lowerbound, upperbound, bkernel, bsctype, bltype)

    config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
    set_kernel!(config, bkernel)  # calls set_kernel of the C API
    config.sc_type = bsctype # maximum a posteriori method
    config.noise = 1.0e-12
    isa(bltype, Number) ? 0 : config.l_type = bltype
    config.n_inner_iterations = 200
    config.n_init_samples = 200
    config.random_seed = 0
    # config.force_jump = 50
    config.verbose_level = 0

    optimizer, optimum = bayes_optimization(errfunBayes, lowerbound, upperbound, config)

    Rcsf = optimizer[1]
    E = optimizer[2]
    P_0 = optimizer[3]

    return Rcsf, E, P_0
end

function getModelBayesPss(lowerbound, upperbound, bkernel, bsctype, bltype)

    config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
    set_kernel!(config, bkernel)  # calls set_kernel of the C API
    config.sc_type = bsctype # maximum a posteriori method
    config.noise = 1.0e-12
    isa(bltype, Number) ? 0 : config.l_type = bltype
    config.n_inner_iterations = 100
    config.n_init_samples = 200
    config.random_seed = 0
    # config.force_jump = 50
    config.verbose_level = 0

    optimizer, optimum = bayes_optimization(errfunBayesPss, lowerbound, upperbound, config)

    Rcsf = optimizer[1]
    E = optimizer[2]
    P_0 = optimizer[3]
    Pss = optimizer[4]

    return Rcsf, E, P_0, Pss
end

function getModelBayesStaticP0(lowerbound, upperbound, bkernel, bsctype, bltype)

    config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
    set_kernel!(config, bkernel)  # calls set_kernel of the C API
    config.sc_type = bsctype # maximum a posteriori method
    config.noise = 1.0e-12
    isa(bltype, Number) ? 0 : config.l_type = bltype
    config.n_inner_iterations = 100
    config.n_init_samples = 200
    config.random_seed = 0
    # config.force_jump = 50
    config.verbose_level = 0

    optimizer, optimum = bayes_optimization(errfunBayesStaticP0, lowerbound, upperbound, config)

    Rcsf = optimizer[1]
    E = optimizer[2]
    Pss = optimizer[3]

    return Rcsf, E, Pss
end

function logit(x, min, max)
    (max - min) * (1 / (1 + exp(-x))) + min
end

function logitt(x)
    (1 / (1 + exp(-x)))
end

function g!(G, X)
    # G[1] = Symbolics.value(substitute(diffRcsf, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2])))
    # G[2] = Symbolics.value(substitute(diffE, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2])))
    # G[3] = Symbolics.value(substitute(diffP0, Dict(I_inf=>Data["I_inf"], P_b=>Data["P_b"], Rcsf=>X[1], P_0=>X[3], E=>X[2])))

    P_b = Data["P_b"]
    I_inf = Data["I_inf"]
    Rcsf = X[1]
    E = X[2]
    P_0 = X[3]
    t = 0

    G[1] = (-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / (I_inf * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) - ((P_0 - P_b) / (Rcsf^2) + E * I_inf * t * ((P_b - P_0) / (Rcsf^2)) * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf))) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2))

    G[2] = I_inf * t * (I_inf + (P_b - P_0) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf))

    G[3] = 1 + ((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / (I_inf * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) - (-1 / Rcsf + (E * I_inf * t * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * t * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2))
end

function h!(H, X)
    P_b = Data["P_b"]
    I_inf = Data["I_inf"]
    Rcsf = X[1]
    E = X[2]
    P_0 = X[3]

    H[1] = ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - ((P_0 - P_b) / (Rcsf^2) + E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) * ((P_b - P_0) / (Rcsf^2) - E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) + (-(P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * ((I_inf * (E^2) * (P_b - P_0) * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^2) - 2Rcsf * ((P_0 - P_b) / (Rcsf^4)) - 2E * I_inf * Rcsf * ((P_b - P_0) / (Rcsf^4)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) + (-2Rcsf * (P_0 - P_b) * ((P_b - P_0) / (Rcsf^4))) / (I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) - ((P_0 - P_b) / (Rcsf^2) + E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2))
    H[2] = (I_inf * (P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * ((P_0 - P_b) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) + (-E * I_inf * (P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * ((P_b - P_0) / (Rcsf^2)) * ((P_0 - P_b) / Rcsf - I_inf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - I_inf * ((P_0 - P_b) / Rcsf - I_inf) * ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - ((P_0 - P_b) / (Rcsf^2) + E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))
    H[3] = (1 / Rcsf + (-E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - ((P_0 - P_b) / (Rcsf^2) + E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) + (-2((P_0 - P_b) / (Rcsf^2))) / (I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) + (-(P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * (1 / (Rcsf^2) + (-E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^2) + (I_inf * (E^2) * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) * ((P_0 - P_b) / (Rcsf^2) + E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))
    H[4] = (-(P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * ((I_inf * (P_b - P_0) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^2) + (E * I_inf * (P_b - P_0) * ((P_0 - P_b) / Rcsf - I_inf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - I_inf * ((P_0 - P_b) / Rcsf - I_inf) * ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) - I_inf * ((P_0 - P_b) / Rcsf - I_inf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * ((P_b - P_0) / (Rcsf^2) - E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))
    H[5] = (-I_inf * (P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * (((P_0 - P_b) / Rcsf - I_inf)^2) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) + (I_inf^2) * (((P_0 - P_b) / Rcsf - I_inf)^2) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * (exp(-E * (I_inf + (P_b - P_0) / Rcsf))^2)
    H[6] = (-(P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * (I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + E * I_inf * ((P_0 - P_b) / Rcsf - I_inf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) / (Rcsf * ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) - I_inf * (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) * ((P_0 - P_b) / Rcsf - I_inf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) - I_inf * ((P_0 - P_b) / Rcsf - I_inf) * (1 / Rcsf + (-E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))
    H[7] = (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - (-1 / Rcsf + (E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) * ((P_b - P_0) / (Rcsf^2) - E * I_inf * ((P_b - P_0) / (Rcsf^2)) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) + (2((P_b - P_0) / (Rcsf^2))) / (I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) + (-(P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * (1 / (Rcsf^2) + (-E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^2) + (I_inf * (E^2) * (P_b - P_0) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf^3))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - (-1 / Rcsf + (E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * ((-(P_b - P_0) * ((P_b - P_0) / (Rcsf^2))) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2))
    H[8] = (-I_inf * (P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf * ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) + (-E * I_inf * (P_b - P_0) * ((P_0 - P_b) / Rcsf - I_inf) * (I_inf + (P_b - P_0) / Rcsf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / (Rcsf * ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) - I_inf * ((P_0 - P_b) / Rcsf - I_inf) * (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - (-1 / Rcsf + (E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))
    H[9] = (1 / Rcsf + (-E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2) - (-1 / Rcsf + (E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf) * (((P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf)) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^4)) * ((2P_b - 2P_0) / Rcsf + 2I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)))) + (2(1 / Rcsf)) / (I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf) + (-I_inf * (E^2) * (P_b - P_0) * (I_inf + (P_b - P_0) / Rcsf) * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / ((Rcsf^2) * ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) - (((P_0 - P_b) / Rcsf + (P_0 - P_b) / Rcsf - I_inf) / ((I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf)) + (P_b - P_0) / Rcsf)^2)) * (-1 / Rcsf + (E * I_inf * exp(-E * (I_inf + (P_b - P_0) / Rcsf))) / Rcsf)

end

function solveDerivatives()
    Symbolics.@variables E P_0 I_inf P_b Rcsf t I_b

    f = P_0 + (((((P_b - P_0) / Rcsf) + I_inf) * (P_b - P_0)) / (((P_b - P_0) / Rcsf) + I_inf * exp(-E * t * (((P_b - P_0) / Rcsf) + I_inf))))
    # First order partiala derivatives
    dRcsf = Differential(Rcsf)
    diffRcsf = expand_derivatives(dRcsf(f))
    dE = Differential(E)
    diffE = expand_derivatives(dE(f))
    dP0 = Differential(P_0)
    diffP0 = expand_derivatives(dP0(f))

    #Second order partial derivatives
    # global H1 = expand_derivatives(dRcsf(diffRcsf))
    # global H2 = expand_derivatives(dRcsf(diffE))
    # global H3 = expand_derivatives(dRcsf(diffP0))

    # global H4 = expand_derivatives(dE(diffRcsf))
    # global H5 = expand_derivatives(dE(diffE))
    # global H6 = expand_derivatives(dE(diffP0))

    # global H7 = expand_derivatives(dP0(diffRcsf))
    # global H8 = expand_derivatives(dP0(diffE))
    # global H9 = expand_derivatives(dP0(diffP0))

    return diffRcsf, diffE, diffP0
end

function plotmodel(I_b, E, P_0, cscheme, fmodel)

    if cscheme == "dark"
        # FOR WEBSITE
        # bgcolor=:transparent
        # fgcolor=:transparent
        # icp_color=:white
        # model_color=parse(Colorant,"#4A212E")
        # inf_color=:white
        # linw = 3;

        bgcolor = RGB(0.13, 0.15, 0.15)
        fgcolor = :white
        icp_color = :cadetblue
        model_color = :orange
        inf_color = RGB(0.15, 0.17, 0.17)
    else
        bgcolor = :white
        fgcolor = :black
        icp_color = :teal
        model_color = :orangered2
        inf_color = :grey95
    end

    icp = Data["ICP"]
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    P_b = Data["P_b"]
    plateau_end = Data["plateau_end"]

    plateau_end > infend ? endidx = plateau_end : endidx = infend

    # vspan([infstart, infend], fillcolor=inf_color, alpha=0.5, linecolor=:transparent, label="Infusion")
    vspan([infstart, infend], fillcolor=inf_color, alpha=0.05, linecolor=:transparent, label="Infusion")

    xtks = LinRange(0, infend, 10)
    xtklabels = round.(collect(xtks) ./ 6, digits=1)

    plot!(icp, color=icp_color, background=bgcolor, lw=2, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, legend=:outertopright, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(firstindex(icp), endidx))

    #FOR WEBSITE
    # plot!(icp, color=icp_color, background=bgcolor, lw=3, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(50, endidx),axis=false,legend=false, dpi=300)

    Pm = zeros(endidx)
    if fmodel == "Pss"
        Pmodel, rmserr = calc_model_plot_Pss(I_b, E)
    elseif fmodel == "4param"
        # Pmodel, rmserr = calc_model_plot_Pss(I_b, E)
    else
        Pmodel, rmserr = calc_model_plot(I_b, E, P_0)
    end

    Pm[firstindex(Pm):infend] .= Pmodel
    Pm[infend+1:end] .= Pm[infend]
    Pm[firstindex(Pm):infstart] .= P_b

    plot!(Pm, c=model_color, label="Model", linestyle=:dash, lw=3)

    title!("I_b = $(round(I_b,digits=2))\n" * "Rcsf = $(round(value(Rcsf),digits=2))\n" * "E = $(round(value(E),digits=2))\n" * "P_0 = $(round(value(P_0),digits=2))\n" * "error = $rmserr", titlealign=:left, titlefontsize=8, xlabel="Time [min]", ylabel="ICP [mmHg]")
end

function plotmodel(I_b, E, P_0, μ, σ, cscheme, fmodel)

    if cscheme == "dark"
        # FOR WEBSITE
        # bgcolor=:transparent
        # fgcolor=:transparent
        # icp_color=:white
        # model_color=parse(Colorant,"#4A212E")
        # inf_color=:white
        # linw = 3;

        bgcolor = RGB(0.13, 0.15, 0.15)
        fgcolor = :white
        icp_color = :cadetblue
        model_color = :orange
        inf_color = RGB(0.15, 0.17, 0.17)
    else
        bgcolor = :white
        fgcolor = :black
        icp_color = :teal
        model_color = :orangered2
        inf_color = :skyblue
    end

    icp = Data["ICP"]
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    P_b = Data["P_b"]
    plateau_end = Data["plateau_end"]

    plateau_end > infend ? endidx = plateau_end : endidx = infend

    # vspan([infstart, infend], fillcolor=inf_color, alpha=0.5, linecolor=:transparent, label="Infusion")
    vspan([infstart, infend], fillcolor=inf_color, alpha=0.1, linecolor=:transparent, label="Infusion")

    xtks = LinRange(0, infend, 10)
    xtklabels = round.(collect(xtks) ./ 6, digits=1)

    plot!(icp, color=icp_color, background=bgcolor, lw=2, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, legend=:outertopright, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(firstindex(icp), endidx))

    #FOR WEBSITE
    # plot!(icp, color=icp_color, background=bgcolor, lw=3, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(50, endidx),axis=false,legend=false, dpi=300)

    # Have to deal with this global
    global Pm = zeros(endidx)
    if fmodel == "Pss"
        Pmodel, rmserr = calc_model_plot_Pss(I_b, E)
    else
        Pmodel, rmserr = calc_model_plot(I_b, E, P_0)
    end

    Pm[firstindex(Pm):infend] .= Pmodel
    Pm[infend+1:end] .= Pm[infend]
    Pm[firstindex(Pm):infstart] .= P_b

    num_iter = 10000
    w, ci = getCI(μ, σ, num_iter)
    plot!(Pm, ribbon=w, c=model_color, fillalpha=0.1, label="Bayes", linestyle=:dash, lw=3)
    title!("I_b = $(round(I_b,digits=2))\n" * "Rcsf = $(round(value(Rcsf),digits=2))\n" * "E = $(round(value(E),digits=2))\n" * "P_0 = $(round(value(P_0),digits=2))\n" * "error = $rmserr", titlealign=:left, titlefontsize=8, xlabel="Time [min]", ylabel="ICP [mmHg]")
end

function plotmodel(I_b, E, P_0, Pss, μ, σ, cscheme, fmodel)

    if cscheme == "dark"
        # FOR WEBSITE
        # bgcolor=:transparent
        # fgcolor=:transparent
        # icp_color=:white
        # model_color=parse(Colorant,"#4A212E")
        # inf_color=:white
        # linw = 3;

        bgcolor = RGB(0.13, 0.15, 0.15)
        fgcolor = :white
        icp_color = :cadetblue
        model_color = :orange
        inf_color = RGB(0.15, 0.17, 0.17)
    else
        bgcolor = :white
        fgcolor = :black
        icp_color = :teal
        model_color = :orangered2
        inf_color = :skyblue
    end

    icp = Data["ICP"]
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    P_b = Data["P_b"]
    plateau_end = Data["plateau_end"]

    plateau_end > infend ? endidx = plateau_end : endidx = infend

    # vspan([infstart, infend], fillcolor=inf_color, alpha=0.5, linecolor=:transparent, label="Infusion")
    vspan([infstart, infend], fillcolor=inf_color, alpha=0.1, linecolor=:transparent, label="Infusion")

    xtks = LinRange(0, infend, 10)
    xtklabels = round.(collect(xtks) ./ 6, digits=1)

    plot!(icp, color=icp_color, background=bgcolor, lw=2, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, legend=:outertopright, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(firstindex(icp), endidx))

    #FOR WEBSITE
    # plot!(icp, color=icp_color, background=bgcolor, lw=3, grid=false, xticks=(xtks, xtklabels), foreground_color=fgcolor, label="ICP", ylims=[minimum(icp) * 0.9, maximum(icp) * 1.1], xlims=(50, endidx),axis=false,legend=false, dpi=300)

    Pm = zeros(endidx)
    Pmodel, rmserr = calc_model_plot(I_b, E, P_0, Pss)

    Pm[firstindex(Pm):infend] .= Pmodel
    Pm[infend+1:end] .= Pm[infend]
    Pm[firstindex(Pm):infstart] .= P_b

    num_iter = 10000
    w, ci = getCI(μ, σ, num_iter)
    plot!(Pm, ribbon=w, c=model_color, fillalpha=0.1, label="Bayes", lw=3)
    # plot!(Pm, ribbon=w, c=:purple, fillalpha=0.1, label="Gradient descent", lw=3)

    title!("I_b = $(round(I_b,digits=2))\n" * "Rcsf = $(round(value(Rcsf),digits=2))\n" * "E = $(round(value(E),digits=2))\n" * "P_0 = $(round(value(P_0),digits=2))\n" * "error = $rmserr", titlealign=:left, titlefontsize=8, xlabel="Time [min]", ylabel="ICP [mmHg]")
end

function getCI(μ, σ, num_iter)
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    icp = Data["ICP"][infstart:infend]
    Pmodel = zeros(infend)
    model_err = zeros(num_iter)
    numvars = length(μ)
    θ = zeros(numvars, num_iter)

    for i = 1:numvars
        d = Normal(μ[i], σ[i])
        θ[i, :] = rand(d, num_iter)
    end

    if numvars == 3
        θ[1, :] = (Data["P_b"] .- θ[3, :]) ./ θ[1, :]
        for j = 1:num_iter
            θᵢ = θ[:, j]
            Pmodel = calc_model_plot(θᵢ[1], θᵢ[2], θᵢ[3], θᵢ[3])[1]
            Pmodel = Pmodel[infstart:end]
            model_err[j] = mean(abs.(Pmodel .- icp))
        end
    elseif numvars==2
        θ[1, :] = (Data["P_b"] .- Data["P0_static"]) ./ θ[1, :]
        for j = 1:num_iter
            θᵢ = θ[:, j]
            Pmodel = calc_model_plot(θᵢ[1], θᵢ[2], Data["P0_static"], Data["P0_static"])[1]
            Pmodel = Pmodel[infstart:end]
            model_err[j] = mean(abs.(Pmodel .- icp))
        end
    else
        θ[1, :] = (Data["P_b"] .- θ[4, :]) ./ θ[1, :]
        for j = 1:num_iter
            θᵢ = θ[:, j]
            Pmodel = calc_model_plot(θᵢ[1], θᵢ[2], θᵢ[3], θᵢ[4])[1]
            Pmodel = Pmodel[infstart:end]
            model_err[j] = mean(abs.(Pmodel .- icp))
        end
    end

    x̂ = mean(model_err)
    s = std(model_err)
    z = 0.95
    n = num_iter

    ci_low = x̂ + z * s / sqrt(n)
    ci_high = x̂ + z * s / sqrt(n)
    y1 = Pmodel .- ci_low
    y2 = Pmodel .+ ci_high
    w = (y2 .- y1) ./ 2

    return w, ci_low
end

function calc_model_plot(I_b, E, P_0)
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - P_0
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend) .+ Data["P_b"]
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm, round(rmserr, digits=4)
end

function calc_model_plot(I_b, E, P_0, Pss)
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - P_0
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend) .+ Data["P_b"]
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm, round(rmserr, digits=4)
end

function calc_model_plot_Pss(I_b, E)
    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    P_0 = Data["P_0"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - P_0
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend)
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm, round(rmserr, digits=5)
end

function solvemodel(Rcsf, E, P_0)
    # function solvemodel(I_b, E, P_0)

    # Rcsf = sigmoid(Rcsf, lb[1], ub[1])
    # E = sigmoid(E, lb[2], ub[2])
    # P_0 = sigmoid(P_0, lb[3], ub[3])
    # I_b = sigmoid(I_b,lb[4],ub[4])

    I_b = (Data["P_b"] - P_0) / Rcsf

    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - P_0
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend) .+ Data["P_b"]
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm, rmserr
end

function solvemodel(Rcsf, E, P_0, Pss)
    # function solvemodel(I_b, E, P_0)

    Rcsf = sigmoid(Rcsf, lb[1], ub[1])
    E = sigmoid(E, lb[2], ub[2])
    P_0 = sigmoid(P_0, lb[3], ub[3])
    Pss = sigmoid(Pss, lb[3], ub[3])
    # I_b = sigmoid(I_b,lb[4],ub[4])

    I_b = (Data["P_b"] - Pss) / Rcsf

    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - Pss
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend)
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    # rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm
end

function solvemodelPss(Rcsf, E, Pss)
    # function solvemodel(I_b, E, P_0)

    Rcsf = sigmoid(Rcsf, lb[1], ub[1])
    E = sigmoid(E, lb[2], ub[2])
    P_0 = Data["P_0"]
    Pss = sigmoid(Pss, lb[3], ub[3])
    # I_b = sigmoid(I_b,lb[4],ub[4])

    I_b = (Data["P_b"] - Pss) / Rcsf

    infstart = Data["infusion_start_frame"]
    infend = Data["infusion_end_frame"]
    I_inf = Data["I_inf"]
    Rn = Data["Rn"]
    ΔP = Data["P_b"] - Pss
    icp = Data["ICP"]
    It = I_b + I_inf
    Pm = zeros(infend)
    errorVal = 0.0

    for i = infstart:infend
        t = (i - infstart) / 6
        y = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
        Pm[i] = y
        errorVal += (icp[i] - y)^2
    end

    # rmserr = 100 * sqrt(errorVal) / length(Pm) / abs(mean(icp[infstart:infend]))
    return Pm
end

# transform sigmoid function
function sigmoid(x, lb, ub)
    (ub - lb) * (1 / (1 + exp(-x))) + lb
end