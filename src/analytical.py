import numpy as np
from scipy.stats import norm

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the analytical Black-Scholes price for a European Call option.
    
    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility (annualized)
        
    Returns:
        float: The option price
    """
    # Sanity check
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Call Price
    # norm.cdf is Cumulative Distribution Function N(x)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

if __name__ == "__main__":
    # Test Case: Deep OTM
    params = {
        "S": 100.0,
        "K": 140.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2
    }
    
    price = black_scholes_call(**params)
    
    print("-" * 30)
    print("BLACK-SCHOLES ANALYTICAL BENCHMARK")
    print("-" * 30)
    print(f"Parameters: {params}")
    print(f"Theoretical Price: {price:.6f}")
    print("-" * 30)