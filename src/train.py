import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from src.conformal import ConformalClassifier


def load_config(config_path="config.yaml") -> dict:
    """Loads configuration parameters from a YAML file."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def build_lr_pipeline(config: dict) -> Pipeline:
    """Builds a Logistic Regression classifier pipeline."""
    num_features = [
        "char_count",
        "word_count",
        "avg_word_length",
        "cap_ratio",
        "exclamation_ratio",
        "question_ratio",
    ]
    num_transformer = StandardScaler()

    tfidf_params = config["tfidf_params"]
    text_feature = "clean_text"
    text_transformer = TfidfVectorizer(
        max_features=tfidf_params["max_features"],
        ngram_range=tuple(tfidf_params["ngram_range"]),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("text", text_transformer, text_feature),
        ]
    )

    lr_params = config["logistic_regression_params"]
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=lr_params["max_iter"],
                    C=lr_params["C"],
                    random_state=config["data_params"]["random_state"],
                ),
            ),
        ]
    )
    return lr_pipeline


def build_xgb_pipeline(config: dict) -> Pipeline:
    """Builds an XGBoost classifier pipeline."""
    num_features = [
        "char_count",
        "word_count",
        "avg_word_length",
        "cap_ratio",
        "exclamation_ratio",
        "question_ratio",
    ]
    num_transformer = StandardScaler()

    tfidf_params = config["tfidf_params"]
    text_feature = "clean_text"
    text_transformer = TfidfVectorizer(
        max_features=tfidf_params["max_features"],
        ngram_range=tuple(tfidf_params["ngram_range"]),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("text", text_transformer, text_feature),
        ]
    )

    xgb_params = config["xgboost_params"]
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=xgb_params["n_estimators"],
                    max_depth=xgb_params["max_depth"],
                    learning_rate=xgb_params["learning_rate"],
                    subsample=xgb_params["subsample"],
                    colsample_bytree=xgb_params["colsample_bytree"],
                    random_state=xgb_params["random_state"],
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    return xgb_pipeline


def evaluate_conformal(conformal_model, X_test, y_test):
    """Computes metrics for the conformal classifier on the test set."""
    pred_sets, _ = conformal_model.predict_set(X_test)
    y_test_arr = y_test.values

    # Empirical Coverage: Is the true label in the prediction set?
    in_set = [y_test_arr[i] in pred_sets[i] for i in range(len(y_test_arr))]
    empirical_coverage = np.mean(in_set)

    # Set sizes
    set_sizes = [len(s) for s in pred_sets]
    avg_set_size = np.mean(set_sizes)

    # Frequency of set sizes
    size_counts = pd.Series(set_sizes).value_counts().to_dict()
    singleton_pct = size_counts.get(1, 0) / len(X_test)

    metrics_dict = {
        "empirical_coverage": empirical_coverage,
        "avg_set_size": avg_set_size,
        "singleton_pct": singleton_pct,
        "size_counts": size_counts,
    }
    return metrics_dict


def run_experiment(model_name, pipeline, X_train, y_train, X_calib, y_calib, X_test, y_test, config):
    """Trains, conformal-calibrates, logs to MLflow, and returns metrics & model."""
    alpha = config["data_params"]["alpha"]
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n--- Running Experiment: {model_name} ---")
        
        # Log common parameters
        mlflow.log_param("model_type", model_name.split("_")[0])
        mlflow.log_param("alpha_significance", alpha)
        mlflow.log_param("target_coverage", 1 - alpha)
        mlflow.log_param("tfidf_max_features", config["tfidf_params"]["max_features"])

        # Log model-specific parameters
        if "LogisticRegression" in model_name:
            lr_params = config["logistic_regression_params"]
            mlflow.log_param("lr_C", lr_params["C"])
            mlflow.log_param("lr_max_iter", lr_params["max_iter"])
        elif "XGBoost" in model_name:
            xgb_params = config["xgboost_params"]
            mlflow.log_param("xgb_n_estimators", xgb_params["n_estimators"])
            mlflow.log_param("xgb_max_depth", xgb_params["max_depth"])
            mlflow.log_param("xgb_learning_rate", xgb_params["learning_rate"])

        # Train baseline
        print("Training baseline estimator...")
        pipeline.fit(X_train, y_train)
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        print(f"Baseline Train Accuracy: {train_acc:.4%}")
        print(f"Baseline Test Accuracy:  {test_acc:.4%}")
        mlflow.log_metric("baseline_train_accuracy", train_acc)
        mlflow.log_metric("baseline_test_accuracy", test_acc)

        # Calibrate conformal prediction
        print(f"Calibrating Conformal Prediction...")
        conformal_model = ConformalClassifier(estimator=pipeline, alpha=alpha)
        conformal_model.fit_calibration(X_calib, y_calib)
        print(f"Calibration complete. Threshold q_hat: {conformal_model.q_hat:.4f}")
        mlflow.log_metric("conformal_q_hat", conformal_model.q_hat)

        # Evaluate conformal metrics
        metrics = evaluate_conformal(conformal_model, X_test, y_test)
        print(f"Empirical Coverage:      {metrics['empirical_coverage']:.2%}")
        print(f"Average Set Size:        {metrics['avg_set_size']:.4f}")
        print(f"Certainty (Singleton %): {metrics['singleton_pct']:.2%}")

        # Log metrics
        mlflow.log_metric("empirical_coverage", metrics["empirical_coverage"])
        mlflow.log_metric("avg_set_size", metrics["avg_set_size"])
        mlflow.log_metric("singleton_pct", metrics["singleton_pct"])

        # Log sklearn model flavor
        mlflow.sklearn.log_model(pipeline, "sklearn_pipeline")
        
        run_results = {
            "model_name": model_name,
            "accuracy": test_acc,
            "conformal_model": conformal_model,
            "metrics": metrics,
        }
        return run_results


def train_and_calibrate():
    # Load settings from config
    config = load_config()
    paths = config["paths"]
    data_params = config["data_params"]

    processed_data_path = Path(paths["processed_data"])
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_data_path}. Run pipeline first.")

    print("Loading processed dataset...")
    df = pd.read_parquet(processed_data_path)

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
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=data_params["test_size"], random_state=data_params["random_state"], stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=data_params["random_state"], stratify=y_temp
    )

    # Save splits to disk
    processed_dir = Path(paths["processed_data"]).parent
    
    # Clean assignments for saving
    train_split = X_train.assign(label=y_train)
    calib_split = X_calib.assign(label=y_calib)
    test_split = X_test.assign(label=y_test)
    
    train_split.to_parquet(Path(paths["train_split"]), index=False)
    calib_split.to_parquet(Path(paths["calibration_split"]), index=False)
    test_split.to_parquet(Path(paths["test_split"]), index=False)
    print("Splits successfully saved to data/processed/")

    # Setup MLflow experiment
    mlflow.set_experiment("Fake_Review_Detection")

    # Define pipelines
    lr_pipeline = build_lr_pipeline(config)
    xgb_pipeline = build_xgb_pipeline(config)

    # Run comparative experiments
    lr_results = run_experiment(
        "LogisticRegression_Conformal", lr_pipeline, 
        X_train, y_train, X_calib, y_calib, X_test, y_test, config
    )
    
    xgb_results = run_experiment(
        "XGBoost_Conformal", xgb_pipeline, 
        X_train, y_train, X_calib, y_calib, X_test, y_test, config
    )

    # Print Comparison Table
    comparison_data = [
        {
            "Model": "Logistic Regression",
            "Accuracy": f"{lr_results['accuracy']:.2%}",
            "Conformal Coverage": f"{lr_results['metrics']['empirical_coverage']:.2%}",
            "Avg Set Size": f"{lr_results['metrics']['avg_set_size']:.4f}",
            "Certainty (Singleton %)": f"{lr_results['metrics']['singleton_pct']:.2%}",
        },
        {
            "Model": "XGBoost",
            "Accuracy": f"{xgb_results['accuracy']:.2%}",
            "Conformal Coverage": f"{xgb_results['metrics']['empirical_coverage']:.2%}",
            "Avg Set Size": f"{xgb_results['metrics']['avg_set_size']:.4f}",
            "Certainty (Singleton %)": f"{xgb_results['metrics']['singleton_pct']:.2%}",
        }
    ]
    df_comparison = pd.DataFrame(comparison_data)
    print("\n================ MODEL COMPARISON ================")
    print(df_comparison.to_string(index=False))
    print("==================================================\n")

    # Determine best model based on accuracy
    best_results = xgb_results if xgb_results["accuracy"] > lr_results["accuracy"] else lr_results
    print(f"Best model: {best_results['model_name']} (Accuracy: {best_results['accuracy']:.2%})")

    # Save best model to disk
    model_dir = Path(paths["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    conformal_model_path = Path(paths["conformal_model"])
    
    with open(conformal_model_path, "wb") as f:
        pickle.dump(best_results["conformal_model"], f)
    
    print(f"Saved best conformal model to: {conformal_model_path.resolve()}")


if __name__ == "__main__":
    train_and_calibrate()
