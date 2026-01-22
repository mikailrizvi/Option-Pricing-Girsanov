import numpy as np
from naive_mc import naive_monte_carlo
from analytical import black_scholes_call

def run_stress_test():
    # Parameters for Deep OTM Call
    params = {
        "S": 100.0, "K": 140.0, "T": 1.0, "r": 0.05, "sigma": 0.2
    }
    
    # Ground Truth
    true_price = black_scholes_call(**params)
    
    print(f"Target Analytical Price: {true_price:.6f}")
    print("-" * 50)
    print(f"{'Run':<5} | {'Est. Price':<12} | {'Error vs Truth':<15} | {'Std Error (Internal)':<15}")
    print("-" * 50)
    
    prices = []
    
    # Run 20 independent batches of 100,000 simulations
    for i in range(20):
        price, se = naive_monte_carlo(**params, num_simulations=100_000)
        prices.append(price)
        
        diff = price - true_price
        print(f"{i+1:<5} | {price:.6f}     | {diff:+.6f}        | {se:.6f}")
        
    print("-" * 50)
    mean_est = np.mean(prices)
    std_est = np.std(prices)
    print(f"Average of runs: {mean_est:.6f}")
    print(f"Std Dev of runs: {std_est:.6f}")

if __name__ == "__main__":
    run_stress_test()