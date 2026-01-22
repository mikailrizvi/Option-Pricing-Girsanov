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