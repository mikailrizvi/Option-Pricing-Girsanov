import numpy as np
import time

# --- IMPORTS ---
try:
    from src.naive_mc import naive_monte_carlo
    from src.analytical import black_scholes_call
    from src.is_mc import importance_sampling_engine
except ImportError:
    from naive_mc import naive_monte_carlo
    from analytical import black_scholes_call
    from is_mc import importance_sampling_engine

def run_benchmark():
    print("\n" + "="*66)
    print("BENCHMARK: Naive Monte Carlo vs. Importance Sampling (Day 6)")
    print("="*66)

    # 1. PARAMETERS
    # S=100, K=140 means the option is usually worthless.
    params = {
        "S0": 100.0,  # Note: Check if your other files use 'S' or 'S0'
        "K": 140.0, 
        "T": 1.0, 
        "r": 0.05, 
        "sigma": 0.2
    }
    
    n_sims = 10_000  
    theta_shift = 1.8 # Shift mean by 1.8 std devs to target K=140

    # 2. GROUND TRUTH
    true_price = black_scholes_call(params["S0"], params["K"], params["T"], params["r"], params["sigma"])
    print(f"Target Analytical Price: {true_price:.6f}\n")

    print(f"{'Method':<20} | {'Est. Price':<10} | {'Std Error':<10} | {'Time (s)':<8}")
    print("-" * 66)

    # 3. RUN NAIVE MONTE CARLO
    start_time = time.time()
    naive_price, naive_se = naive_monte_carlo(
        params["S0"], params["K"], params["T"], params["r"], params["sigma"], n_sims
    )
    naive_time = time.time() - start_time
    print(f"{'Naive MC':<20} | {naive_price:.6f}   | {naive_se:.6f}   | {naive_time:.4f}")

    # 4. RUN IMPORTANCE SAMPLING
    start_time = time.time()
    is_price, is_se = importance_sampling_engine(
        params["S0"], params["K"], params["T"], params["r"], params["sigma"], n_sims, theta_shift
    )
    is_time = time.time() - start_time
    print(f"{'Importance Samp':<20} | {is_price:.6f}   | {is_se:.6f}   | {is_time:.4f}")

    # 5. THE VERDICT
    print("-" * 66)
    vr_factor = (naive_se**2) / (is_se**2)
    
    print(f"\nRESULTS ANALYSIS:")
    print(f"1. Naive Error Range: +/- {naive_se * 1.96:.4f} (95% Confidence)")
    print(f"2. IS Error Range:    +/- {is_se * 1.96:.4f} (95% Confidence)")
    print(f"3. Variance Reduction Factor: {vr_factor:.1f}x")
    
    print(f"\nCONCLUSION:")
    if vr_factor > 1.0:
        print(f"SUCCESS: Importance Sampling is {vr_factor:.1f} times more efficient.")
        print(f"Needed {int(n_sims * vr_factor):,} Naive simulations to match this precision.")
    else:
        print("ADJUSTMENT NEEDED")

if __name__ == "__main__":
    run_benchmark()