"""
Vespa CLI - Command Line Interface for AI Tasks

This script provides a command-line interface (CLI) for managing AI-related tasks 
such as training models, making predictions, and listing available models or dataset formats.

Usage:
    - Train a model:
        python cli.py train --task detection --model yolo --dataset path/to/dataset --format yolo
    
    - Make a prediction:
        python cli.py predict --task classification --model resnet --input path/to/image.jpg
    
    - List available models:
        python cli.py list-models
    
    - List supported dataset formats:
        python cli.py list-dataset-formats

Commands:
    - `train` : Trains a model with a specified dataset and format.
    - `predict` : Runs inference on an input image using a selected model.
    - `list-models` : Displays the available models for different tasks.
    - `list-dataset-formats` : Shows the supported dataset formats.

Arguments:
    - `--task` : Specifies the AI task type (`classification`, `regression`, `detection`).
    - `--dataset` : Path to the dataset used for training.
    - `--format` : Format of the dataset (`yolo`, `coco`, `pascal_voc`).
    - `--model` : Name of the model to be used for training or inference.
    - `--input` : Path to an input image for prediction.

Dependencies:
    - Requires the `vespa` library for AI model handling.

Author:
    - Vespa AI Team
"""

import sys
import os

# Adjust system path to include the Vespa module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from vespa import vespa as vp

def main():
    parser = argparse.ArgumentParser(description="Vespa - CLI for AI Tasks")
    
    parser.add_argument("command", choices=["train", "predict", "list-models", "list-dataset-formats"],
                        help="Command to execute")

    parser.add_argument("--task", choices=["classification", "regression", "detection"], 
                        help="Specify the task type")
    
    parser.add_argument("--dataset", type=str, help="Path to the dataset for training")
    parser.add_argument("--format", choices=["yolo", "coco", "pascal_voc"], help="Dataset format")
    
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("--input", type=str, help="Path to input image for inference")

    args = parser.parse_args()

    if args.command == "train":
        if not args.task or not args.format or not args.model or not args.dataset:
            print("Error: Missing parameters for training.")
            return
        vp.set_task(args.task)
        vp.set_dataset_format(args.format)
        vp.set_model(args.model)
        result = vp.train(args.dataset)
        print("Training Result:", result)

    elif args.command == "predict":
        if not args.task or not args.model or not args.input:
            print("Error: Missing parameters for prediction.")
            return
        vp.set_task(args.task)
        vp.set_model(args.model)
        output = vp.predict(args.input)
        print("Prediction Output:", output)

    elif args.command == "list-models":
        models = vp.list_models()
        print("Available Models:", models)
    
    elif args.command == "list-dataset-formats":
        formats = vp.list_dataset_formats()
        print("Supported Dataset Formats:", formats)

if __name__ == "__main__":
    main()
