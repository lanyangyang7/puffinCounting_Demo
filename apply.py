import os
import math
from ultralytics import YOLO

MODEL_PATH = "result/detect/runs/puffin_counting/weights/best.pt"

TEST_IMAGE_DIR = "dataset/images/test"
TEST_LABEL_DIR = "dataset/labels/test"

CONF_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

# Metrics
total_relative_error = 0
total_absolute_error = 0
total_squared_error = 0

num_images = 0

image_extensions = [".jpg", ".jpeg", ".png"]

for image_name in os.listdir(TEST_IMAGE_DIR):

    if not any(image_name.endswith(ext) for ext in image_extensions):
        continue

    image_path = os.path.join(TEST_IMAGE_DIR, image_name)

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(TEST_LABEL_DIR, label_name)

    # Ground Truth Count
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            gt_count = len(f.readlines())
    else:
        gt_count = 0

    # Prediction
    results = model.predict(
        image_path,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    pred_count = len(results[0].boxes)

    # Errors
    absolute_error = abs(pred_count - gt_count)
    squared_error = (pred_count - gt_count) ** 2

    # Relative Error
    if gt_count > 0:
        relative_error = absolute_error / gt_count
    else:
        relative_error = 0 if pred_count == 0 else 1

    # Accumulate
    total_relative_error += relative_error
    total_absolute_error += absolute_error
    total_squared_error += squared_error

    num_images += 1

    print("=" * 50)
    print(f"Image: {image_name}")
    print(f"Ground Truth Count : {gt_count}")
    print(f"Predicted Count    : {pred_count}")
    print(f"Absolute Error     : {absolute_error}")
    print(f"Relative Error     : {relative_error:.4f}")

# Final Metrics
mean_relative_error = total_relative_error / num_images
mae = total_absolute_error / num_images
mse = total_squared_error / num_images
rmse = math.sqrt(mse)

print("\n")
print("=" * 50)
print("Final Counting Metrics")
print("=" * 50)

print(f"Mean Relative Counting Error : {mean_relative_error:.4f}")
print(f"MAE                          : {mae:.4f}")
print(f"MSE                          : {mse:.4f}")
print(f"RMSE                         : {rmse:.4f}")