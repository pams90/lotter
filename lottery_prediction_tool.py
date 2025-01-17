
# File: lottery_prediction_tool.py
# Purpose: Reusable lottery prediction tool

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LotteryPredictionTool:
    def __init__(self, historical_data, num_simulations=10000):
        self.historical_data = historical_data
        self.num_simulations = num_simulations
        self.features = None
        self.rankings = None

    def preprocess_data(self):
        """Prepare features and labels from historical data."""
        # Extract numbers and create metrics (historical frequency, pairs, etc.)
        number_columns = [f"Number {i}" for i in range(1, 7)]
        all_numbers = pd.concat([self.historical_data[col] for col in number_columns])
        self.features = pd.DataFrame({
            "Historical Frequency": all_numbers.value_counts(normalize=True).sort_index()
        })

    def calculate_pair_trends(self):
        """Calculate pair trends from historical data."""
        pair_frequencies = Counter(
            pair for draw in self.historical_data.values for pair in combinations(draw, 2)
        )
        pair_scores = pd.Series(0, index=self.features.index)
        for (num1, num2), freq in pair_frequencies.items():
            pair_scores[num1] += freq
            pair_scores[num2] += freq
        self.features["Pair Score"] = pair_scores / pair_scores.max()

    def generate_rankings(self, weights):
        """Generate rankings based on weighted metrics."""
        self.rankings = self.features.copy()
        self.rankings["Final Score"] = (
            weights["historical"] * self.rankings["Historical Frequency"] +
            weights["pair"] * self.rankings.get("Pair Score", 0)
        )
        self.rankings.sort_values(by="Final Score", ascending=False, inplace=True)

    def simulate_draws(self):
        """Perform Monte Carlo simulations based on rankings."""
        probabilities = self.rankings["Final Score"] / self.rankings["Final Score"].sum()
        simulated_draws = [
            np.random.choice(self.rankings.index, size=6, replace=False, p=probabilities)
            for _ in range(self.num_simulations)
        ]
        return pd.Series(np.concatenate(simulated_draws)).value_counts(normalize=True)

    def display_rankings(self, top_n=10):
        """Display top-ranked numbers."""
        return self.rankings.head(top_n)

# Example usage
if __name__ == "__main__":
    # Placeholder for historical data
    historical_data = pd.DataFrame({
        "Number 1": np.random.randint(1, 50, 1000),
        "Number 2": np.random.randint(1, 50, 1000),
        "Number 3": np.random.randint(1, 50, 1000),
        "Number 4": np.random.randint(1, 50, 1000),
        "Number 5": np.random.randint(1, 50, 1000),
        "Number 6": np.random.randint(1, 50, 1000),
    })

    # Initialize and preprocess
    tool = LotteryPredictionTool(historical_data)
    tool.preprocess_data()
    tool.calculate_pair_trends()

    # Generate rankings
    weights = {"historical": 0.7, "pair": 0.3}
    tool.generate_rankings(weights)

    # Display results
    print("Top 10 Numbers:")
    print(tool.display_rankings())

    # Perform simulations
    simulated_frequencies = tool.simulate_draws()
    print("\nSimulated Frequencies:")
    print(simulated_frequencies.head(10))
