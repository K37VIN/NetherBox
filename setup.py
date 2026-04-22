from setuptools import find_packages,setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("-e")
    ]

setup(
    name="automl-pipeline",
    version="0.1.0",
    author="K37VIN",
    author_email="kardev321@gmail.com",
    description="NetherBox-AutoML pipeline with MLflow, DVC, Docker, and GitHub Actions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "automl-train=src.pipline.training_pipeline:TrainingPipeline",
        ]
    },
)
