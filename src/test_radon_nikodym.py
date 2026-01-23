import numpy as np
from girsanov_utils import calculate_likelihood_ratio

def test_measure_change():
    print("-" * 30)
    print("RADON-NIKODYM SANITY CHECK")
    print("-" * 30)
    
    # 1. Setup
    N = 1_000_000
    lam = 1.53  # The approximate shift calculation
    
    print(f"Testing with N={N:,} paths and lambda={lam}")
    
    # 2. Simulate the "Shifted" World
    # Step A: Generate standard random noise
    Z_raw = np.random.standard_normal(N)
    
    # Step B: Apply the shift
    Z_shifted = Z_raw + lam
    
    # 3. Calculate Weights
    L = calculate_likelihood_ratio(Z_shifted, lam)
    
    # 4. Verification
    mean_L = np.mean(L)
    std_L = np.std(L)
    
    print(f"Mean of Weights: {mean_L:.6f}  (Target: 1.000000)")
    print(f"StdDev of Weights: {std_L:.6f}")
    
    # Strict check (allowing for tiny random noise)
    if abs(mean_L - 1.0) < 0.005:
        print("\n>> PASS: Measure change preserves probability.")
    else:
        print("\n>> FAIL: The weights do not average to 1.0.")

if __name__ == "__main__":
    test_measure_change()