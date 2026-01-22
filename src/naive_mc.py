import numpy as np
import time

# UPDATE: Return type is now a tuple (float, float)
def naive_monte_carlo(S: float, K: float, T: float, r: float, sigma: float, num_simulations: int = 1_000_000) -> tuple[float, float]:
    """
    Prices a European Call option using Naive Monte Carlo simulation.
    
    Returns:
        (price, standard_error)
    """
    
    # 1. Generate random noise (Z)
    Z = np.random.standard_normal(num_simulations)
    
    # 2. Simulate terminal stock prices (S_T)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)
    
    # 3. Calculate Payoffs
    payoffs = np.maximum(S_T - K, 0)
    
    # 4. Discount back to present value
    # We discount the payoffs first to get the PV of every single path
    discount_factor = np.exp(-r * T)
    discounted_payoffs = payoffs * discount_factor
    
    # 5. Calculate Price and Standard Error
    price = np.mean(discounted_payoffs)
    std_dev = np.std(discounted_payoffs)
    standard_error = std_dev / np.sqrt(num_simulations)
    
    return price, standard_error

if __name__ == "__main__":
    params = {
        "S": 100.0,
        "K": 140.0,  # Deep OTM
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2
    }
    
    N_SIMS = 1_000_000
    
    print("-" * 30)
    print("NAIVE MONTE CARLO DIAGNOSTICS")
    print("-" * 30)
    
    start_time = time.time()
    price, se = naive_monte_carlo(**params, num_simulations=N_SIMS)
    end_time = time.time()
    
    print(f"Simulations:    {N_SIMS:,.0f}")
    print(f"Estimated Price: {price:.6f}")
    print(f"Standard Error:  {se:.6f}")
    print(f"95% Conf Int:    [{price - 1.96*se:.6f}, {price + 1.96*se:.6f}]")
    print(f"Time Elapsed:    {end_time - start_time:.4f} seconds")
    print("-" * 30)