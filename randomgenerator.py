import numpy as np
import pandas as pd

# Regime constants
BULL = 0
SIDEWAYS = 1
BEAR = 2

transition_matrix = np.array([
    [0.98, 0.015, 0.005], # Bull
    [0.03, 0.95,  0.02],  # Sideways
    [0.02, 0.05,  0.93]   # Bear
])

annual_returns = np.array([0.12, 0.01, -0.35])
annual_volatilities = np.array([0.12, 0.15, 0.35])

def generate_price_path(start_price: float, start_regime: int, num_years: int, num_simulations: int, random_seed: int = None):
    """
    generate_price_path generates simulated price paths based on a regime-switching model (vectorized version).
    
    :param start_price: starting price of the asset
    :param start_regime: starting market regime (0: Bull, 1: Sideways, 2: Bear)
    :param num_years: number of years to simulate
    :param num_simulations: number of simulation paths to generate
    :param random_seed: random seed for reproducibility
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Assume 252 trading days per year
    trading_days = num_years * 252
    
    # Pre-generate all regime transitions for all simulations
    # Shape: (num_simulations, trading_days)
    regimes = np.zeros((num_simulations, trading_days), dtype=int)
    regimes[:, 0] = start_regime
    
    # Generate regime transitions vectorized across all simulations
    for day in range(1, trading_days):
        # Get current regimes for all simulations
        prev_regimes = regimes[:, day - 1]
        
        # Vectorized regime selection using cumulative probabilities
        rand_vals = np.random.rand(num_simulations)
        cumsum_probs = np.cumsum(transition_matrix[prev_regimes], axis=1)
        
        # Determine next regime for all simulations at once
        regimes[:, day] = (rand_vals[:, None] > cumsum_probs).sum(axis=1)
    
    # Convert annual parameters to daily based on regimes (vectorized)
    daily_mean_returns = annual_returns[regimes] / 252
    daily_volatilities = annual_volatilities[regimes] / np.sqrt(252)
    
    # Generate random daily returns from normal distribution N(μ, σ)
    # Each day's return is a random variable drawn from the regime's distribution
    # Shape: (num_simulations, trading_days)
    daily_returns = np.random.normal(daily_mean_returns, daily_volatilities)
    
    
    
    # Initialize price paths matrix
    # Shape: (num_simulations, trading_days + 1)
    price_paths = np.zeros((num_simulations, trading_days + 1))
    price_paths[:, 0] = start_price
    
    # Calculate all prices using exponential (to ensure prices stay positive)
    price_paths[:, 1:] = start_price * np.cumprod(1 + daily_returns, axis=1)
    
    return price_paths
