# Define the likelihood function
# This function calculates the likelihood of the model given the data and the parameters
function likelihood(params, icp_inf, alpha)
  # Calculate the predicted values of the model
  y_pred = model(params)

  # Calculate the sum of squared errors
  sse = sum((y_pred .- icp_inf) .^ 2)

  if alpha < 1.0
    sse_pv = press_vol_curve(params[1], params[3], icp_inf)[5]
    sse_pv *= 100 # the are not the same order of magnitude - this is heuristic and needs to be revisited
    sse_total = alpha * sse + (1.0 - alpha) * sse_pv
    # println(@sprintf("PV error %0.2f", sse_pv))
    # println(@sprintf("Fitting error %0.2f", sse))
  else
    sse_total = sse
  end

  # Return the likelihood of the model given the data and the parameters
  return sse_total
end

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
  icp = Data["ICP"]
  It = I_b + I_inf
  Pm = zeros(infend - infstart + 1)
  errorVal = 0.0

  for i = infstart:infend
    t = (i - infstart) / 6
    Pm[i-infstart+1] = It * ΔP / (I_b + (I_inf * exp(-E * It * t))) + P_0 + (I_inf * Rn)
  end

  return Pm
end

# Define the acceptance probability function
# This function calculates the probability of accepting a proposed new state
# based on the current state, the proposed new state, and the physiologically
# defined ranges for the parameters
function acceptance_probability(current, proposed, ranges, data, alpha)
  # Check if any of the proposed parameter values are outside of the defined ranges

  Ib = (Data["P_b"] - proposed[3]) / proposed[1]
  if any(proposed .< ranges[:, 1]) || any(proposed .> ranges[:, 2]) || Ib >= 1.0 || Ib <= 0.0
    # If any of the proposed values are outside of the defined ranges, return 0
    return 0
  else
    # If all of the proposed values are within the defined ranges,
    # return the probability of accepting the proposed new state
    # based on the current state and the proposed new state
    current = likelihood(current, data, alpha)
    proposed = likelihood(proposed, data, alpha)
    return exp((current - proposed) / 2)
  end
end

# Define the Metropolis-Hastings algorithm
function metropolis_hastings(data, means, stddevs, ranges, num_samples, alpha)
  # Initialize the Markov chain with the starting point
  # The starting point is the mean of the prior distributions
  chain = zeros(num_samples, 3)
  chisave = zeros(num_samples)
  chain[1, :] = means

  # Run the Markov chain for the specified number of samples

  for i in 2:num_samples
    # Sample a proposed new state from the normal distributions centered at the current state
    current = chain[i-1, :]
    proposed = randn(3) .* stddevs .+ current
    # Calculate the acceptance probability of the proposed new state
    p = acceptance_probability(current, proposed, ranges, data, alpha)

    chisave[i] = likelihood(current, data, alpha) # Undesirable global assignment

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

# Calculate the means and standard deviations of the posterior distributions
# of the fitted parameters
function mean_and_stddev(chain)
  params_modes = zeros(3)

  for i = 1:3
    smooth_dist = kde(chain[:, i], bandwidth=0.01) # Mode is unstable if there are very small differences between values - smoothen
    smooth_dist_vals = collect(smooth_dist.x)
    params_modes[i] = smooth_dist_vals[findmax(smooth_dist.density)[2]]
    params_modes[i] = median(chain[:, i])
    # params_modes[i] = StatsBase.mode(chain[:,i])
  end

  # params_modes[i] = median(chain[:,])

  params_means = mean(chain, dims=1)
  params_stddevs = std(chain, dims=1)
  return params_modes, params_means, params_stddevs
end

# Define the main function
function main(fileID, num_samples, priors, alpha)
  # Load the data
  datapath = "/Users/jjc/CSF/Recordings/"
  path = pwd()
  savepath = "/Users/jjc/CSF/"
  files = glob("*.hdf5", datapath)
  j = fileID
  filename = files[j]

  global Data = readCSF(filename)
  infstart = Data["infusion_start_frame"]
  infend = Data["infusion_end_frame"]
  global icp_inf = Data["ICP"][infstart:infend]

  # Specify parameter ranges
  # TODO: add I_b
  lowerbound = [0.01, 0.01, -10.0]
  upperbound = [50.0, 1.0, Data["P_b"]]
  ranges = hcat(lowerbound, upperbound)

  # Define the starting point of the Markov chain
  if priors == "informative"
    means = [15.5, 0.18, 2.8]
    stddevs = [10.36, 0.14, 10.54]
  else
    means = zeros(3) .+ 0.01
    stddevs = zeros(3) .+ 0.1
  end

  # Run the Metropolis-Hastings algorithm for the specified number of samples
  burnin = Int64(round(0.2 * num_samples, digits=0))
  chain, chisave = metropolis_hastings(icp_inf, means, stddevs, ranges, num_samples, alpha)
  chain = chain[burnin:end, :]
  Ib_chain = (Data["P_b"] .- chain[:, 3]) ./ chain[:, 1]
  return chain, chisave, Ib_chain
end
