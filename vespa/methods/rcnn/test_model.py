import os
import torch
import yaml
from vespa.datasets.dataset_factory import DatasetFactory
from vespa.datasets.yolo.yolo_transforms import get_yolo_test_transforms, get_yolo_train_transforms
from vespa.methods.rcnn.model import RCNN
import logging

def normalize_path(path):
    """
    Normaliza os caminhos para evitar problemas relacionados ao sistema operacional.
    """
    return os.path.normpath(path)

def main():
    # Configuration
    root_dir = normalize_path("//192.168.155.240/Robotica/CME/dataset_cme_v4/laparoscopia_06-2024/tools")
    train_file = "train.txt"
    val_file = "val.txt"
    test_file = "test.txt"
    yaml_file = os.path.join(root_dir, "tools.yaml")
    batch_size = 4
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load class mapping from YAML file
    with open(yaml_file, "r") as f:
        data_config = yaml.safe_load(f)
    class_mapping = {int(k): v for k, v in data_config["names"].items()}
    num_classes = len(class_mapping)

    # # Create datasets
    # def validate_dataset_file(dataset_type, root_dir, txt_file):
    #     txt_path = normalize_path(os.path.join(root_dir, txt_file))
    #     if not os.path.exists(txt_path):
    #         raise FileNotFoundError(f"Dataset file {txt_path} not found.")
    #     with open(txt_path, 'r') as f:
    #         for line in f:
    #             image_path = normalize_path(os.path.join(root_dir, line.strip()))
    #             label_path = image_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg', '.txt')
    #             if not os.path.exists(image_path):
    #                 logging.warning(f"Image file {image_path} not found.")
    #             if not os.path.exists(label_path):
    #                 logging.warning(f"Label file {label_path} not found.")

    # logging.info("Validating training dataset...")
    # validate_dataset_file("yolo", root_dir, train_file)

    # logging.info("Validating validation dataset...")
    # validate_dataset_file("yolo", root_dir, val_file)

    # logging.info("Validating test dataset...")
    # validate_dataset_file("yolo", root_dir, test_file)

    train_dataset = DatasetFactory.create_dataset(
        dataset_type="yolo",
        root_dir=root_dir,
        txt_file=train_file,
        image_size=1088,
        transforms=get_yolo_train_transforms(),
    )

    val_dataset = DatasetFactory.create_dataset(
        dataset_type="yolo",
        root_dir=root_dir,
        txt_file=val_file,
        image_size=1088,
        transforms=get_yolo_test_transforms(),
    )

    test_dataset = DatasetFactory.create_dataset(
        dataset_type="yolo",
        root_dir=root_dir,
        txt_file=test_file,
        image_size=1088,
        transforms=get_yolo_test_transforms(),
    )

    # Initialize model
    model = RCNN(num_classes=num_classes, weights="DEFAULT", optimizer_name="adam")

    # Train the model
    logging.info("Starting training...")
    model.train_model(train_dataset, batch_size=batch_size, epochs=epochs, device=device)

    # Validate the model
    logging.info("Evaluating on validation set...")
    val_loss = model.validate_model(val_dataset, batch_size=batch_size, device=device)
    logging.info(f"Validation Loss: {val_loss:.4f}")

    # Test the model
    logging.info("Testing the model...")
    test_metrics = model.test_model(test_dataset, batch_size=batch_size, device=device, class_mapping=class_mapping)
    print(f"Metrics: {test_metrics}")
  
    # Save the model
    model_save_path = normalize_path(os.path.join(root_dir, "rcnn_model.pth"))
    model.save_model(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # Print additional model details
    logging.info("Model Summary:")
    model.print_model_summary()

    trainable_params = model.count_trainable_parameters()
    logging.info(f"Trainable Parameters: {trainable_params}")

if __name__ == "__main__":
    main()
