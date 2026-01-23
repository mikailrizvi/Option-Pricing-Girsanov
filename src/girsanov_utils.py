import numpy as np

def calculate_drift_shift(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the optimal drift adjustment (lambda) to center the 
    simulation around the strike price K.
    
    This forces the simulation to sample 'interesting' paths that end up
    near the strike price, rather than wasting time on deep OTM paths.
    
    Formula:
    lambda = [ ln(K/S) - (r - 0.5*sigma^2)T ] / (sigma * sqrt(T))
    """
    
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        return 0.0
        
    numerator = np.log(K / S) - (r - 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)
    
    return numerator / denominator

if __name__ == "__main__":
    # Test with standard parameters
    params = {
        "S": 100.0,
        "K": 140.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2
    }
    
    lam = calculate_drift_shift(**params)
    
    print("-" * 30)
    print("DRIFT SHIFT CALCULATOR")
    print("-" * 30)
    print(f"Parameters: {params}")
    print(f"Optimal Shift (lambda): {lam:.6f}")
    print("-" * 30)
    print("--------------------------------------------")
    print(f"shift our random draws (Z) by +{lam:.6f}")
    print("standard deviations to make the stock hit $140 on average.")

def calculate_likelihood_ratio(Z_shifted: np.ndarray, lam: float) -> np.ndarray:
    """
    Calculates the Radon-Nikodym derivative (Likelihood Ratio) L.
    
    Formula:
    L = exp( -lambda * Z_shifted + 0.5 * lambda^2 )
    
    Parameters:
        Z_shifted (np.array): The random numbers used in the simulation (already shifted!)
        lam (float): The shift amount (lambda)
        
    Returns:
        np.array: The weights for each path.
    """
    # Vectorized calculation
    exponent = -lam * Z_shifted + 0.5 * lam**2
    L = np.exp(exponent)
    
    return L