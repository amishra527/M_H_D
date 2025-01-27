from ultralytics import YOLO
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



model = YOLO("/best_7.pt") 

inference_results = model("KIIT-MiTA/test/images", device=0, verbose=False)

def inference_vis(res, n_ims, rows):
    cols = n_ims // rows
    plt.figure(figsize=(20, 10))
    for idx, r in enumerate(res):
        if idx == n_ims:
            break
        plt.subplot(rows, cols, idx + 1)
        or_im_rgb = np.array(Image.open(r.path).convert("RGB"))
        if idx == n_ims:
            break
        for i in r:
            for bbox in i.boxes:
                box = bbox.xyxy[0]
                x1, y1, x2, y2 = box
                coord1, coord2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(or_im_rgb, coord1, coord2, color=(255, 0, 0), thickness=2)
        plt.imshow(or_im_rgb)
        plt.title(f"Image#{idx + 1}")
        plt.axis("off")


inference_vis(res=inference_results, n_ims=15, rows=3)

# Evaluate the model's performance on the validation set
results = model.val()
print(results)
