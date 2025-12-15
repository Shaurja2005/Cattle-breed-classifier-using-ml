# Cattle & Buffalo Breed Classifier

This project provides a deep learning-based image classifier for identifying various breeds of cattle and buffalo using PyTorch and transfer learning. The model leverages a pre-trained ResNet-18 architecture, fine-tuned to distinguish between multiple breeds from a curated dataset.

## Features

- **Multi-class Classification**: Classifies images into 12 distinct cattle and buffalo breeds.
- **Transfer Learning**: Uses a pre-trained ResNet-18 model for efficient training on smaller datasets.
- **Data Augmentation**: Implements resizing, flipping, and normalization to improve model robustness.
- **Training Pipeline**: Includes a complete training loop with accuracy reporting and model saving.

## Supported Breeds

The model is designed to classify the following breeds:
- Brown Swiss
- Deoni
- Gir
- Holstein Friesian
- Jaffrabadi
- Kangayam
- Kankrej
- Khillari
- Murrah
- Pandharpuri
- Sahiwal
- Toda

## Dataset Structure

The project expects the dataset to be organized in the standard ImageFolder format:

```
dataset/
├── train/
│   ├── Brown_Swiss/
│   ├── Deoni/
│   └── ... (other breeds)
├── valid/
│   ├── Brown_Swiss/
│   └── ...
└── test/
    ├── Brown_Swiss/
    └── ...
```

## Requirements

- Python 3.x
- PyTorch
- torchvision

You can install the necessary dependencies using pip:

```bash
pip install torch torchvision
```

## Usage

1. **Prepare Data**: Ensure your images are placed in the `train`, `valid`, and `test` directories as described above.

2. **Train the Model**: Run the training script to fine-tune the model.

   ```bash
   python train.py
   ```

   The script will train the model for 10 epochs (default) and print the training loss and accuracy for each epoch.

3. **Output**: After training, the model weights will be saved to `cow_breed_classifier.pth`.

## Model Details

- **Architecture**: ResNet-18 (Pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Optimizer**: Adam (Learning Rate: 0.001)
- **Loss Function**: CrossEntropyLoss
## Results

| Metric | Accuracy |
| :--- | :--- |
| **Validation Accuracy** | *84.054%* |
| **Test Accuracy** | *77.16%* |

## License

[MIT License](LICENSE)
