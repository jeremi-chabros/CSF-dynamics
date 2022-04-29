# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Julia 1.7.2
#     language: julia
#     name: julia-1.7
# ---

# +
# parsexml.jl
using EzXML, CSV, Glob
global selections, results

datapath = "/Users/jjc/Documents/SSC/CSF Data/selected_recordings"
path = pwd();
savepath = "/Users/jjc/CSF/"
files = glob("*.xml", datapath);

for filename in files
    
data = readxml(filename)

icm_data = root(data)

vars = elements(icm_data)

SingleAnalysis = firstelement(vars[2])
SingleAnalysis = elements(SingleAnalysis)
Variables = elements(SingleAnalysis[1])
Selections = elements(SingleAnalysis[2])
Parameters = elements(SingleAnalysis[3])
Results = elements(SingleAnalysis[4])

inf_rate = Parameters[1]["Value"]

# Selections
global selections = Dict{String, String}

for i in 1:length(Selections)
    sel = elements(Selections[i])
    sel_name = Selections[i]["Name"]
    sel_st = sel[1]["StartTime"]
    sel_en = sel[1]["EndTime"]
    
    global selections =  merge!(selections, Dict(sel_name*" from " => sel_st*" to "*sel_en))
end

global results = Dict{String, Float64}

for i in 1:length(Results)
    res = elements(Results[i]);
    res_name = Results[i]["Name"]
    res_val = Results[i]["Value"]

    global results =  merge!(results, Dict(res_name => res_val))
end
println(savepath*filename[length(datapath)+2:end-4]*"_results.csv")
CSV.write(savepath*filename[length(datapath)+2:end-4]*"_results.csv", results)
CSV.write(savepath*filename[length(datapath)+2:end-4]*"_selections.csv", selections)
end
# -




