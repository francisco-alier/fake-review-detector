import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from src.conformal import ConformalClassifier


def build_pipeline(max_features=5000, C=1.0) -> Pipeline:
    """
    Builds a scikit-learn pipeline that combines text TF-IDF and scaled numerical features.
    """
    num_features = [
        "char_count",
        "word_count",
        "avg_word_length",
        "cap_ratio",
        "exclamation_ratio",
        "question_ratio",
    ]
    num_transformer = StandardScaler()

    text_feature = "clean_text"
    text_transformer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("text", text_transformer, text_feature),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, C=C, random_state=42),
            ),
        ]
    )
    return pipeline


def evaluate_conformal_predictions(conformal_model, X_test, y_test, alpha):
    """
    Computes performance metrics for the conformal classifier on the test set.
    """
    pred_sets, probs = conformal_model.predict_set(X_test)
    y_test_arr = y_test.values

    # Empirical Coverage: Is the true label in the prediction set?
    in_set = [y_test_arr[i] in pred_sets[i] for i in range(len(y_test_arr))]
    empirical_coverage = np.mean(in_set)

    # Set sizes
    set_sizes = [len(s) for s in pred_sets]
    avg_set_size = np.mean(set_sizes)

    # Frequency of set sizes
    size_counts = pd.Series(set_sizes).value_counts().to_dict()

    evaluation_metrics = {
        "empirical_coverage": empirical_coverage,
        "avg_set_size": avg_set_size,
        "size_counts": size_counts,
    }
    return evaluation_metrics


def train_and_calibrate():
    data_path = Path("data/processed/fake_reviews_processed.parquet")
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run pipeline first.")

    print("Loading processed dataset...")
    df = pd.read_parquet(data_path)

    # Prepare features and target
    X = df[
        [
            "clean_text",
            "char_count",
            "word_count",
            "avg_word_length",
            "cap_ratio",
            "exclamation_ratio",
            "question_ratio",
        ]
    ]
    y = df["label"]

    print("Splitting dataset (70% Train, 15% Calibration, 15% Test)...")
    # First split off the training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Split the remaining 30% equally into Calibration and Test
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train size:       {len(X_train):,}")
    print(f"Calibration size: {len(X_calib):,}")
    print(f"Test size:        {len(X_test):,}")

    # Save splits for downstream evaluation notebooks (cleaner ML pipeline)
    print("Saving data splits to data/processed/...")
    
    # Bundle features and targets together for the parquet files
    train_split = X_train.assign(label=y_train)
    calib_split = X_calib.assign(label=y_calib)
    test_split = X_test.assign(label=y_test)
    
    train_split.to_parquet(processed_dir / "train.parquet", index=False)
    calib_split.to_parquet(processed_dir / "calibration.parquet", index=False)
    test_split.to_parquet(processed_dir / "test.parquet", index=False)

    # Define hyperparameters for this experiment
    max_features = 5000
    C_param = 1.0
    alpha = 0.05

    # Start MLflow Run
    mlflow.set_experiment("Fake_Review_Detection")
    with mlflow.start_run():
        print("Logging experiment metrics and artifacts to MLflow...")
        # Log parameters
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C_regularization", C_param)
        mlflow.log_param("alpha_significance", alpha)
        mlflow.log_param("target_coverage", 1 - alpha)

        # Step 1: Train the underlying ML Pipeline
        print("Training scikit-learn baseline model...")
        pipeline = build_pipeline(max_features=max_features, C=C_param)
        pipeline.fit(X_train, y_train)

        # Evaluate baseline point accuracy
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        print(f"Baseline Train Accuracy: {train_acc:.4%}")
        print(f"Baseline Test Accuracy:  {test_acc:.4%}")

        # Log baseline metrics
        mlflow.log_metric("baseline_train_accuracy", train_acc)
        mlflow.log_metric("baseline_test_accuracy", test_acc)

        # Step 2: Calibrate the Conformal Predictor
        print(f"Calibrating conformal prediction with alpha={alpha}...")
        conformal_model = ConformalClassifier(estimator=pipeline, alpha=alpha)
        conformal_model.fit_calibration(X_calib, y_calib)
        print(f"Conformal Calibration complete. Threshold q_hat: {conformal_model.q_hat:.4f}")
        mlflow.log_metric("conformal_q_hat", conformal_model.q_hat)

        # Step 3: Evaluate Conformal Metrics
        print("Evaluating conformal prediction on test set...")
        metrics = evaluate_conformal_predictions(conformal_model, X_test, y_test, alpha)

        print("\n--- Conformal Prediction Evaluation ---")
        print(f"Empirical Coverage on Test:    {metrics['empirical_coverage']:.2%}")
        print(f"Average Prediction Set Size:   {metrics['avg_set_size']:.4f}")

        # Log conformal metrics
        mlflow.log_metric("empirical_coverage", metrics["empirical_coverage"])
        mlflow.log_metric("avg_set_size", metrics["avg_set_size"])
        for size, count in metrics["size_counts"].items():
            pct = count / len(X_test)
            mlflow.log_metric(f"set_size_{size}_pct", pct)
            print(f"  Size {size} (set of {size} labels): {count:,} ({pct * 100:.2f}%)")

        # Step 4: Save Model Artifacts locally
        model_output_path = models_dir / "conformal_model.pkl"
        with open(model_output_path, "wb") as f:
            pickle.dump(conformal_model, f)
        print(f"Saved calibrated conformal model to {model_output_path.resolve()}")

        # Log model object as an artifact in MLflow
        mlflow.log_artifact(str(model_output_path))
        
        # Log the scikit-learn pipeline directly via MLflow flavor
        mlflow.sklearn.log_model(pipeline, "sklearn_pipeline")
        print("MLflow experiment logging complete!")


if __name__ == "__main__":
    train_and_calibrate()
