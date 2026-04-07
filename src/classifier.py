import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ImageClassifier:
    def __init__(self, model_name="simple_cnn", input_shape=(32, 32, 3), num_classes=10):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        if self.model_name == "simple_cnn":
            model = keras.Sequential([
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ])
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        print(f"Training {self.model_name} model...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        print(f"Evaluating {self.model_name} model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, X_new):
        return self.model.predict(X_new)

if __name__ == "__main__":
    # Dummy data for demonstration
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    classifier = ImageClassifier(input_shape=X_train.shape[1:], num_classes=10)
    classifier.train(X_train[:1000], y_train[:1000], epochs=1)
    classifier.evaluate(X_test[:200], y_test[:200])
