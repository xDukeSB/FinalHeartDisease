"""Train, evaluate, compare models, and save pipelines and metadata."""

import argparse
import os
import sys

from src.training import train_and_evaluate_all_models


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Heart Disease prediction models.")
    parser.add_argument("--data", default="data/full_body_heart_checkup_synthetic_3000.csv", help="Path to input CSV dataset.")
    parser.add_argument("--output", default="models", help="Directory to save model artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set fraction.")
    parser.add_argument("--tune", action="store_true", default=True, help="Enable hyperparameter tuning on training partition.")

    args = parser.parse_args()

    # Fallback to local filename if relative data/ not found
    if not os.path.exists(args.data) and os.path.exists("full_body_heart_checkup_synthetic_3000.csv"):
        args.data = "full_body_heart_checkup_synthetic_3000.csv"

    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found at '{args.data}'", file=sys.stderr)
        sys.exit(1)

    print("================================================================")
    print(" ❤️ Heart Disease Prediction — Training & Evaluation Pipeline ")
    print("================================================================")
    print(f"Dataset path: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed} | Held-out test split: {args.test_size * 100:.0f}%")
    print(f"Hyperparameter tuning: {'Enabled (GridSearchCV on train folds)' if args.tune else 'Disabled'}\n")

    print("[1/3] Loading dataset, checking columns, and splitting data...")
    _trained_pipelines, metadata = train_and_evaluate_all_models(
        data_path=args.data,
        output_dir=args.output,
        random_state=args.seed,
        test_size=args.test_size,
        tune_hyperparameters=args.tune
    )

    print(f"\n[2/3] Training completed across {metadata['train_samples']} training samples.")
    print(f"      Evaluating on {metadata['test_samples']} held-out test samples.\n")

    # Display comparison table
    table_rows = []
    for item in metadata["comparison_table"]:
        table_rows.append([
            item["Model"],
            f"{item['Accuracy']:.2f}%",
            f"{item['Precision']:.2f}%",
            f"{item['Recall']:.2f}%",
            f"{item.get('Specificity', 0.0):.2f}%",
            f"{item['F1 Score']:.2f}%",
            f"{item.get('ROC-AUC', 0.0):.2f}%"
        ])

    headers = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "ROC-AUC"]
    try:
        from tabulate import tabulate
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
    except ImportError:
        # Fallback standard printing
        header_str = f"{headers[0]:<20} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<12} | {headers[5]:<10} | {headers[6]:<10}"
        print(header_str)
        print("-" * len(header_str))
        for row in table_rows:
            print(f"{row[0]:<20} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | {row[4]:<12} | {row[5]:<10} | {row[6]:<10}")

    print("\n[3/3] Artifacts saved:")
    print(f"  -> Best model: '{metadata['best_model_name']}'")
    print(f"  -> Pipeline:   '{os.path.join(args.output, 'best_model.joblib')}'")
    print(f"  -> Metadata:   '{os.path.join(args.output, 'model_metadata.json')}'")
    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
