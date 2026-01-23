import numpy as np

def importance_sampling_engine(S0, K, T, r, sigma, n_sims, theta):
    """
    Runs Importance Sampling Monte Carlo.
    Returns: (price_estimate, standard_error)
    """
    # 1. Sample from the shifted distribution Z ~ N(theta, 1)
    Z_star = np.random.normal(theta, 1, n_sims)
    
    # 2. Calculate Asset Price ST
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_star)
    
    # 3. Calculate Payoff
    payoff = np.maximum(ST - K, 0)
    
    # 4. Calculate Likelihood Ratio (The Weight) to un-bias the result
    # Weight = f(x) / g(x) = exp(-theta * Z + 0.5 * theta^2)
    weight = np.exp(-theta * Z_star + 0.5 * theta**2)
    
    # 5. Discounted Weighted Payoff
    discounted_weighted_payoff = np.exp(-r * T) * (payoff * weight)
    
    # 6. Statistics
    price_est = np.mean(discounted_weighted_payoff)
    std_error = np.std(discounted_weighted_payoff) / np.sqrt(n_sims)
    
    return price_est, std_error