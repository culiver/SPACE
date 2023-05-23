import argparse
import json
import numpy as np
from scipy.stats import pearsonr


class CorrelationCalculator:
    def __init__(self, file1_path, file2_path):
        # Load the lists from the JSON files
        with open(file1_path, 'r') as f:
            self.list1 = json.load(f)

        with open(file2_path, 'r') as f:
            self.list2 = json.load(f)

        # Convert the lists to NumPy arrays
        self.array1 = np.array(self.list1)
        self.array2 = np.array(self.list2)

    def calculate_correlation(self):
        # Calculate the Pearson correlation coefficient
        corr, _ = pearsonr(self.array1, self.array2)

        # Print the result
        print(f"Pearson correlation coefficient: {corr:.4f}")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Pearson correlation coefficient from two JSON files')
    parser.add_argument('--f1', type=str, help='Path to the first JSON file')
    parser.add_argument('--f2', type=str, help='Path to the second JSON file')
    args = parser.parse_args()

    # Create a CorrelationCalculator object and calculate the correlation
    calculator = CorrelationCalculator(args.f1, args.f2)
    calculator.calculate_correlation()
