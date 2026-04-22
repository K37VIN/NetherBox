import argparse
import sys

from src.pipline.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run AutoML Training Pipeline")

    parser.add_argument("--data", type=str, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, help="Target column name")

    args = parser.parse_args()

   
    if args.data and args.target:
        dataset_path = args.data
        target_column = args.target
    else:
        if sys.stdin.isatty():
            dataset_path = input("Enter path to your CSV file: ")
            target_column = input("Enter the target column name: ")
        else:
            raise ValueError(
                "In non-interactive mode, you must provide --data and --target arguments"
            )

    pipeline = TrainingPipeline(
        dataset_path=dataset_path,
        target_column=target_column
    )

    result = pipeline.run()

    print("\n── Results ──────────────────────────────")
    print(f"  Best Model   : {result['best_model']}")
    print(f"  Problem Type : {result['problem_type']}")
    print(f"  CV Score     : {result['best_score']}")
    print(f"  Test Metrics : {result['test_metrics']}")
    print("\nAll model scores:")

    for model, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        print(f"  {model:<25} {score:.4f}")


if __name__ == "__main__":
    main()
