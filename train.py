from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")  # option: https://docs.ultralytics.com/zh/models

    # training
    results = model.train(
        # https://docs.ultralytics.com/zh/modes/train
        # base
        data="dataset.yaml",
        epochs=50,
        patience=10,
        imgsz=640,
        batch=16,
        device=0,
        project="result",
        name="puffin_counting",
        pretrained=True,

        # augmentation
        degrees=5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10
    )

    # test set evaluation
    test_metrics = model.val(
        data="dataset.yaml",
        split="test"
    )

    print("\n========== Test Metrics ==========")
    print(f"mAP50: {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall: {test_metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()