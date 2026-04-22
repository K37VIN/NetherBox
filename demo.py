from src.pipline.training_pipeline import TrainingPipeline

dataset_path = input("Enter path to your CSV file:")
target_column = input("Enter the target column name:")

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
print(f"\nAll model scores:")
for model, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
    print(f"  {model:<25} {score:.4f}")