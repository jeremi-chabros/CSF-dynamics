# Define the likelihood function
# This function calculates the likelihood of the model given the data and the parameters
function likelihood(params, icp_inf, alpha, method)
  # Calculate the predicted values of the model
  if method == "standard"
    y_pred = model(params)
    if alpha < 1.0
      sse_pv = press_vol_curve(params[1], params[3], icp_inf)[5]
    end
  elseif method == "Pss"
    y_pred = model_Pss(params)
    if alpha < 1.0
      sse_pv = press_vol_curve(params[1], params[3], icp_inf)[5]
    end
  elseif method == "2"
    y_pred = model(hcat(params..., Data["P0_static"]))
    if alpha < 1.0
      sse_pv = press_vol_curve(params[1], Data["P0_static"], icp_inf)[5]
    end
  else
    y_pred = model_4(params)
    if alpha < 1.0
      sse_pv = press_vol_curve(params[1], params[4], icp_inf)[5]
    end
  end

  # Calculate the sum of squared errors
  sse = sum((y_pred .- icp_inf) .^ 2)

  if alpha < 1.0
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
  P_0 = Data["P0_static"] # Reference ICP

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
function acceptance_probability(current, proposed, ranges, data, alpha, method, stddevs)
  # Check if any of the proposed parameter values are outside of the defined ranges
  if method == "4"
    Ib = (Data["P_b"] - proposed[4]) / proposed[1]
  elseif method == "2"
    Ib = (Data["P_b"] - Data["P0_static"]) / proposed[1]
  else
    Ib = (Data["P_b"] - proposed[3]) / proposed[1]
  end

  # If any of the proposed values are outside of the defined ranges, return 0
  if any(proposed .< ranges[:, 1]) || any(proposed .> ranges[:, 2]) || (Ib >= 1.0) || (Ib <= 0.0)
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
  elseif method == "2"
    num_params = 2
  else
    num_params = 3
  end

  chain = zeros(num_samples, num_params)
  chisave = zeros(num_samples)

  if num_params == 2
    chain[1, :] = means[1:2]
    stddevs = stddevs[1:2]
    ranges = ranges[1:2, :]
  else
    chain[1, :] = means
  end


  # Run the Markov chain for the specified number of samples
  accepted_count = 0.0
  for i in 2:num_samples
    # Sample a proposed new state from the normal distributions centered at the current state
    current = chain[i-1, :]
    proposed = randn(num_params) .* stddevs .+ current

    # Calculate the acceptance probability of the proposed new state
    p = acceptance_probability(current, proposed, ranges, data, alpha, method, stddevs)

    # Save the likelihood trace/chain
    chisave[i] = likelihood(current, data, alpha, method)

    # Accept the proposed new state with the calculated probability
    if rand() < p
      # If the proposed state is accepted, append it to the Markov chain
      chain[i, :] .= proposed
      accepted_count += 1
    else
      # If the proposed state is not accepted, append the current state again
      chain[i, :] .= current
    end
  end

  # Return the Markov chain after running for the specified number of samples
  acceptance_rate = accepted_count/num_samples
  return chain, chisave, acceptance_rate
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
    # params_modes[i] = StatsBase.mode(chain[:,i])
  end
  params_medians = median(chain, dims=1)
  params_means = mean(chain, dims=1)
  params_stddevs = std(chain, dims=1)
  return params_modes, params_medians, params_means, params_stddevs
end

# ---------------------------------------------------------------------------------------

# Define the main function that loads the data and runs MCMC
function main(filename, num_samples, priors, alpha, method, means, stddevs)

  global Data = readCSF(filename)
  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  global icp_inf = Data["ICP"][infstart:infend]

  icp = Data["ICP"]
  amp = Data["AMP"]
  factor = 2 # std from which residuals will be removed
  P0_static, R2_P0 = denoising(icp, amp, factor)
  Data["P0_static"] = P0_static

  # Define the starting point of the Markov chain
  if priors != "informative"
    means = zeros(3) .+ 0.01
    stddevs = zeros(3) .+ 0.1
  end

  # Specify parameter ranges
  if method == "4"
    lowerbound = [0.01, 0.01, -10.0, -10.0]
    upperbound = [50.0, 1.0, Data["P_b"], Data["P_b"]]
    means = [10.0, 0.37, 11.1, 11.1] # Based on literature
    stddevs = [4.95, 0.13, 2.97, 2.97]
  else
    lowerbound = [0.01, 0.01, -10.0]
    upperbound = [50.0, 1.0, Data["P_b"]]
  end
  ranges = hcat(lowerbound, upperbound)

  # Run the Metropolis-Hastings algorithm for the specified number of samples
  burnin = Int64(round(0.2 * num_samples, digits=0))
  chain, chisave, acceptance_rate = metropolis_hastings(icp_inf, means, stddevs, ranges, num_samples, alpha, method)
  chain = chain[burnin:end, :]

  if method == "4"
    Ib_chain = (Data["P_b"] .- chain[:, 4]) ./ chain[:, 1]
  elseif method == "2"
    Ib_chain = (Data["P_b"] .- Data["P0_static"]) ./ chain[:, 1]
  else
    Ib_chain = (Data["P_b"] .- chain[:, 3]) ./ chain[:, 1]
  end
  return chain, chisave, acceptance_rate, Ib_chain, P0_static, R2_P0
end

function denoising(icp, amp, factor)

  icp = Data["ICP"]
  st = Data["infusion_start_frame"]
  en = Data["infusion_end_frame"]

  Pm = icp[st:en]

  trans_st = Data["transition_start_frame"]
  trans_en = Data["transition_end_frame"]
  icp = Data["ICP"][trans_st:trans_en]
  amp = Data["AMP"][trans_st:trans_en]
  rm = .~isnan.(amp)
  rm = rm .&& amp .> 0.1
  amp = amp[rm]
  icp = icp[rm]
  x = icp
  y = amp
  reg = lm(@formula(y ~ x), DataFrame(x=icp, y=amp))
  R2_old = r2(reg)
  residuals = y .- GLM.predict(reg, DataFrame(x=icp, y=amp))
  std_res = residuals ./ std(residuals)
  keep = abs.(std_res) .<= 1
  reg_cleaned = lm(@formula(y ~ x), DataFrame(x=icp[keep], y=amp[keep]))
  c, a = GLM.coeftable(reg_cleaned).cols[1, 1]
  R2 = r2(reg_cleaned)
  y_intercept = -c / a

  return y_intercept, R2
end

