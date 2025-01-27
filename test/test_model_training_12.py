import os, cv2, yaml, random, numpy as np
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from torchvision import transforms as T
from ultralytics import YOLO

import yaml
import wandb
import mlflow
from mlflow import log_params, log_metric
from datetime import datetime
from ultralytics import YOLO  # Ensure you're importing the YOLO model
import optuna  # For hyperparameter optimization

import yaml
import wandb
import mlflow
from mlflow import log_params, log_metric
from datetime import datetime
from ultralytics import YOLO  # Ensure you're importing the YOLO model
import optuna  # For hyperparameter optimization

# Define root directory (update as per your file structure)
root = "/military_object_dataset"

# Load YAML content into a Python dictionary
yml_file_path = f"{root}/military_dataset.yaml"

with open(yml_file_path, "r") as file:
    data = yaml.safe_load(file)

# Update the paths
data["train"] = f"{root}/train/images"
data["val"] = f"{root}/val/images"
data["test"] = f"{root}/test/images"

# Save the updated YAML content to a file
output_path = "updated_config_1.yml"
with open(output_path, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

# Initialize W&B project
wandb.init(project="yolo-hyperparameter-tuning-12")

# MLflow tracking setup
mlflow.set_tracking_uri("http://X.X.X.X:8087")  # Your MLflow tracking URI
experiment_name = "YOLO Hyperparameter Tuning"
mlflow.set_experiment(experiment_name)


# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 10, 30, step=10)
    imgsz = trial.suggest_categorical("imgsz", [320, 480])
    batch = trial.suggest_int("batch", 4, 8, step=4)
    learning_rate = trial.suggest_loguniform("lr0", 1e-4, 1e-2)
    workers = trial.suggest_int("workers", 4, 8, step=4)

    # Initialize the YOLO model
    model = YOLO("yolo11n.pt")  # Replace with the correct YOLO version and weights

    with mlflow.start_run(run_name=f"YOLO_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters to MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("imgsz", imgsz)
        mlflow.log_param("batch", batch)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("workers", workers)
        mlflow.log_param("data", "updated_config_1.yml")

        # Start training
        train_results = model.train(
            data="updated_config_1.yml",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=[0],
            lr0=learning_rate
        )

        # Extract and log the evaluation metric (e.g., mAP)
        metrics = train_results.box.map  # Adjust based on YOLO's output
        print(f"Trial metrics: {metrics}")

        # Log metrics to MLflow
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

        # Log W&B metrics
        wandb.log({"mAP": metrics})

    # Return the primary metric for optimization
    return metrics  # Adjust based on your metric of interest


# Run Optuna study
study = optuna.create_study(direction="maximize")  # Maximize mAP
study.optimize(objective, n_trials=4)

# Best trial results
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")

# Save study results
with open("optuna_study_results_1.yml", "w") as file:
    yaml.dump(study.best_params, file)

# Finish W&B run
wandb.finish()