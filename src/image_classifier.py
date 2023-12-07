# CV-Image-Classifier - Advanced Module

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedModelTrainer:
    """A class to handle advanced model training and evaluation."""
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        logging.info(f"Initialized RandomForestClassifier with n_estimators={n_estimators}, max_depth={max_depth}")

    def load_data(self, filepath):
        """Loads data from a CSV file."""
        try:
            data = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully from {filepath}. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            return None

    def preprocess_data(self, data, target_column):
        """Preprocesses the data by handling missing values and splitting features/target."""
        if data is None:
            return None, None, None, None
        
        data = data.dropna() # Simple handling of missing values
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        X = pd.get_dummies(X, drop_first=True)
        
        logging.info(f"Data preprocessed. Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def train_model(self, X_train, y_train):
        """Trains the RandomForest model."""
        logging.info("Starting model training...")
        self.model.fit(X_train, y_train)
        logging.info("Model training completed.")

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model and prints metrics."""
        logging.info("Starting model evaluation...")
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        logging.info(f"Model Evaluation:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}")
        return accuracy, precision, recall, f1

    def run_pipeline(self, filepath, target_column):
        """Runs the full data loading, preprocessing, training, and evaluation pipeline."""
        data = self.load_data(filepath)
        X, y = self.preprocess_data(data, target_column)
        
        if X is None or y is None:
            logging.error("Failed to preprocess data. Exiting pipeline.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

if __name__ == '__main__':
    dummy_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 5, 100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    dummy_data.to_csv('dummy_data.csv', index=False)

    trainer = AdvancedModelTrainer()
    trainer.run_pipeline('dummy_data.csv', 'target')
