import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"All seeds set to {seed}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract comparison dataset from IMDB names")
    parser.add_argument("--data_file", type=str, default="imdb_actors_actresses.csv",
                        help="Path to the original dataset")
    parser.add_argument("--comparison_size", type=int, default=40000,
                        help="Number of names to extract for comparison (between 10000 and 50000)")
    parser.add_argument("--output_file", type=str, default="comparison_dataset.csv",
                        help="Path to save the comparison dataset")
    parser.add_argument("--training_file", type=str, default="training_dataset.csv",
                        help="Path to save the remaining training dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set all seeds for reproducibility
    set_all_seeds(args.seed)

    # Load the data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} records")

    # Validate comparison size
    if args.comparison_size < 10000 or args.comparison_size > 50000:
        print(f"Warning: Comparison size {args.comparison_size} outside recommended range (10000-50000)")
        print(f"Proceeding anyway...")

    if args.comparison_size > len(df) // 2:
        print(f"Warning: Comparison size {args.comparison_size} is more than half the dataset.")
        print(f"This may leave insufficient data for training.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Extract the comparison dataset (stratified by gender)
    comparison_set, training_set = train_test_split(
        df,
        test_size=len(df) - args.comparison_size,
        random_state=args.seed,
        stratify=df['gender']
    )

    # Save the datasets
    comparison_set.to_csv(args.output_file, index=False)
    training_set.to_csv(args.training_file, index=False)

    # Print statistics
    print("\nDataset Summary:")
    print(f"Original dataset: {len(df)} records")
    print(f"Comparison dataset: {len(comparison_set)} records saved to {args.output_file}")
    print(f"Training dataset: {len(training_set)} records saved to {args.training_file}")

    # Show gender distribution
    print("\nGender Distribution:")
    print("Comparison dataset:")
    gender_counts = comparison_set['gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(comparison_set)) * 100
        print(f"  {gender}: {count} ({percentage:.2f}%)")

    print("\nTraining dataset:")
    gender_counts = training_set['gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(training_set)) * 100
        print(f"  {gender}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
