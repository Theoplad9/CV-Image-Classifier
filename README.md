# CV-Image-Classifier

A Python-based computer vision project for image classification using Convolutional Neural Networks (CNNs) with TensorFlow/Keras.

## Overview

This repository provides a framework for building and training image classification models using Convolutional Neural Networks (CNNs). It includes data loading, preprocessing, model definition, training, and evaluation scripts, primarily utilizing TensorFlow and Keras.

## Features

-   **CNN Architectures:** Implementations of various CNN models (e.g., LeNet, VGG, ResNet-like structures).
-   **Data Augmentation:** Techniques to expand training datasets and improve model generalization.
-   **Transfer Learning:** Support for fine-tuning pre-trained models on new datasets.
-   **Evaluation Metrics:** Reports on accuracy, precision, recall, and F1-score.

## Installation

```bash
git clone https://github.com/Theoplad9/CV-Image-Classifier.git
cd CV-Image-Classifier
pip install -r requirements.txt
```

## Usage

```python
from src.classifier import ImageClassifier

# Initialize classifier with a model (e.g., 'simple_cnn')
classifier = ImageClassifier(model_name='simple_cnn')

# Load and preprocess data (example with dummy data)
# X_train, y_train, X_test, y_test = load_data()
# classifier.train(X_train, y_train, epochs=10)
# accuracy = classifier.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy}")
```

## Project Structure

```
CV-Image-Classifier/
├── data/
│   └── images/
│   └── labels.csv
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   └── classifier.py
├── notebooks/
│   └── model_exploration.ipynb
├── tests/
│   └── test_classifier.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
