import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions():
    # Parameters
    S0, K, T, r, sigma = 100, 140, 1.0, 0.05, 0.2
    n_sims = 10000
    theta = 1.8  
    
    # 1. Standard Normal Draws (Naive)
    Z_naive = np.random.normal(0, 1, n_sims)
    ST_naive = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_naive)
    
    # 2. Shifted Normal Draws (Importance Sampling)
    Z_is = np.random.normal(theta, 1, n_sims)
    ST_is = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_is)
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot Standard Distribution
    sns.histplot(ST_naive, color="blue", label="Standard MC (Naive)", kde=True, stat="density", alpha=0.3)
    
    # Plot Importance Sampling Distribution
    sns.histplot(ST_is, color="red", label=f"Importance Sampling (Shifted)", kde=True, stat="density", alpha=0.3)
    
    # Add a line for the Strike Price
    plt.axvline(K, color='green', linestyle='--', linewidth=2, label=f"Strike Price (K={K})")
    
    plt.title("Visualizing the 'Cheat': Standard vs. Shifted Distributions", fontsize=14)
    plt.xlabel("Stock Price at Maturity ($S_T$)")
    plt.ylabel("Density of Simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Plot generated! Check the popup window.")
    plt.show()

if __name__ == "__main__":
    plot_distributions()