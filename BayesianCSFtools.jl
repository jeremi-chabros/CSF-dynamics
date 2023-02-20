# Define the likelihood function
# This function calculates the likelihood of the model given the data and the parameters
function likelihood(params, icp_inf, alpha, method)
  # Calculate the predicted values of the model
  if method == "standard"
    y_pred = model(params)
  elseif method == "Pss"
    y_pred = model_Pss(params)
  else
    y_pred = model_4(params)
  end

  # Calculate the sum of squared errors
  sse = sum((y_pred .- icp_inf) .^ 2)
  if alpha < 1.0
    sse_pv = press_vol_curve(params[1], params[3], icp_inf)[5]
    sse_pv *= 100 # the are not the same order of magnitude - this is heuristic and needs to be revisited
    sse_total = alpha * sse + (1.0 - alpha) * sse_pv
  else
    sse_total = sse
  end
  # Return the likelihood of the model given the data and the parameters
  return sse_total
end

# ---------------------------------------------------------------------------------------

# Define Marmarou model
function model(params)
  Rcsf = params[1] # Resistance to csf outflow
  E = params[2] # Brain elastance coefficient
  P_0 = params[3] # Reference ICP

  I_b = (Data["P_b"] - P_0) / Rcsf # CSF formation rate

  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  I_inf = Data["I_inf"]

  Rn = Data["Rn"] # Needle resistance (one-needle)
  ΔP = Data["P_b"] - P_0
  It = I_b + I_inf
  Pm = zeros(infend - infstart + 1)

  for i = infstart:infend
    t = (i - infstart) / 6
    Pm[i-infstart+1] = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
  end

  return Pm
end

# ---------------------------------------------------------------------------------------

# Define Marmarou model with static P_0 (optimising over Pss)
function model_Pss(params)
  Rcsf = params[1] # Resistance to csf outflow
  E = params[2] # Brain elastance coefficient
  Pss = params[3] # Pressure in saggital sinus
  P_0 = Data["P_0"] # Reference ICP

  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  I_inf = Data["I_inf"]

  Rn = Data["Rn"] # Needle resistance (one-needle)
  ΔP = Data["P_b"] - P_0
  I_b = (Data["P_b"] - Pss) / Rcsf # CSF formation rate
  It = I_b + I_inf
  Pm = zeros(infend - infstart + 1)

  for i = infstart:infend
    t = (i - infstart) / 6
    Pm[i-infstart+1] = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
  end

  return Pm
end

# ---------------------------------------------------------------------------------------

# Define Marmarou model with 4 free parameters (both P_0 and Pss optimised)
function model_4(params)
  Rcsf = params[1] # Resistance to csf outflow
  E = params[2] # Brain elastance coefficient
  P_0 = params[3] # Reference ICP
  Pss = params[4] # Pressure in saggital sinus

  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  I_inf = Data["I_inf"]

  Rn = Data["Rn"] # Needle resistance (one-needle)

  I_b = (Data["P_b"] - Pss) / Rcsf # CSF formation rate
  It = I_b + I_inf
  Pm = zeros(infend - infstart + 1)

  for i = infstart:infend
    t = (i - infstart) / 6
    Pm[i-infstart+1] = It * (I_b * Rcsf + Pss - P_0) / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
  end

  return Pm
end

# ---------------------------------------------------------------------------------------

# Define the acceptance probability function
# This function calculates the probability of accepting a proposed new state
# based on the current state, the proposed new state, and the physiologically
# defined ranges for the parameters
function acceptance_probability(current, proposed, ranges, data, alpha, method)
  # Check if any of the proposed parameter values are outside of the defined ranges
  if method == "4"
    Ib = (Data["P_b"] - proposed[4]) / proposed[1]
  else
    Ib = (Data["P_b"] - proposed[3]) / proposed[1]
  end
  if any(proposed .< ranges[:, 1]) || any(proposed .> ranges[:, 2]) || Ib >= 1.0 || Ib <= 0.0
    # If any of the proposed values are outside of the defined ranges, return 0
    return 0
  else
    # If all of the proposed values are within the defined ranges,
    # return the probability of accepting the proposed new state
    # based on the current state and the proposed new state
    current = likelihood(current, data, alpha, method)
    proposed = likelihood(proposed, data, alpha, method)
    return exp((current - proposed) / 2)
  end
end

# ---------------------------------------------------------------------------------------

# Define the Metropolis-Hastings algorithm
function metropolis_hastings(data, means, stddevs, ranges, num_samples, alpha, method)
  # Initialize the Markov chain with the starting point
  # The starting point is the mean of the prior distributions

  if method == "4"
    num_params = 4
  else
    num_params = 3
  end

  chain = zeros(num_samples, num_params)
  chisave = zeros(num_samples)
  chain[1, :] = means

  # Run the Markov chain for the specified number of samples

  for i in 2:num_samples
    # Sample a proposed new state from the normal distributions centered at the current state
    current = chain[i-1, :]
    proposed = randn(num_params) .* stddevs .+ current

    # Calculate the acceptance probability of the proposed new state
    p = acceptance_probability(current, proposed, ranges, data, alpha, method)

    # Save the likelihood trace/chain
    chisave[i] = likelihood(current, data, alpha, method)

    # Accept the proposed new state with the calculated probability
    if rand() < p
      # If the proposed state is accepted, append it to the Markov chain
      chain[i, :] .= proposed
    else
      # If the proposed state is not accepted, append the current state again
      chain[i, :] .= current
    end
  end

  # Return the Markov chain after running for the specified number of samples
  return chain, chisave
end

# ---------------------------------------------------------------------------------------

# Calculate the means and standard deviations of the posterior distributions
# of the fitted parameters
function mean_and_stddev(chain)
  num_params = size(chain)[2]
  params_modes = zeros(num_params)
  for i = 1:num_params
    smooth_dist = kde(chain[:, i], bandwidth=0.01) # Mode is unstable if there are very small differences between values - smoothen
    smooth_dist_vals = collect(smooth_dist.x)
    params_modes[i] = smooth_dist_vals[findmax(smooth_dist.density)[2]]
    params_modes[i] = median(chain[:, i])
    # params_modes[i] = StatsBase.mode(chain[:,i])
  end

  params_means = mean(chain, dims=1)
  params_stddevs = std(chain, dims=1)
  return params_modes, params_means, params_stddevs
end

# ---------------------------------------------------------------------------------------

# Define the main function that loads the data and runs MCMC
function main(fileID, num_samples, priors, alpha, method)
  # Load the data
  datapath = "/Users/jjc/CSF/Recordings/"
  # path = pwd()
  # savepath = "/Users/jjc/CSF/"
  files = glob("*.hdf5", datapath)
  j = fileID
  filename = files[j]

  global Data = readCSF(filename)
  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  global icp_inf = Data["ICP"][infstart:infend]

  # Define the starting point of the Markov chain
  if priors == "informative"
    means = [15.5, 0.18, 2.8]
    stddevs = [10.36, 0.14, 10.54]
  else
    means = zeros(3) .+ 0.01
    stddevs = zeros(3) .+ 0.1
  end

  # Specify parameter ranges
  if method == "4"
    lowerbound = [0.01, 0.01, -10.0, -10.0]
    upperbound = [50.0, 1.0, Data["P_b"], Data["P_b"]]
    means = [15.5, 0.18, 2.8, 2.8]
    stddevs = [10.36, 0.14, 10.54, 10.54]
  else
    lowerbound = [0.01, 0.01, -10.0]
    upperbound = [50.0, 1.0, Data["P_b"]]
  end
  ranges = hcat(lowerbound, upperbound)

  # Run the Metropolis-Hastings algorithm for the specified number of samples
  burnin = Int64(round(0.2 * num_samples, digits=0))
  chain, chisave = metropolis_hastings(icp_inf, means, stddevs, ranges, num_samples, alpha, method)
  chain = chain[burnin:end, :]

  if method == "4"
    Ib_chain = (Data["P_b"] .- chain[:, 4]) ./ chain[:, 1]
  else
    Ib_chain = (Data["P_b"] .- chain[:, 3]) ./ chain[:, 1]
  end
  return chain, chisave, Ib_chain
end
