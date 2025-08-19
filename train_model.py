import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_collector import DataCollector

# Import appropriate detector
try:
    from sign_language_detector import SignLanguageDetector
except ImportError:
    from fallback_detector import FallbackSignLanguageDetector as SignLanguageDetector

class ModelTrainer:
    def __init__(self):
        self.collector = DataCollector()
        self.detector = SignLanguageDetector()
        self.scaler = StandardScaler()
    
    def prepare_data(self, test_size=0.2):
        """Load and prepare data for training"""
        print("Loading dataset...")
        X, y, class_names = self.collector.load_dataset()
        
        if len(X) == 0:
            print("No data found! Please collect data first using data_collector.py")
            return None, None, None, None, None
        
        print(f"Dataset loaded: {len(X)} samples, {len(class_names)} classes")
        print(f"Classes: {class_names}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalize the data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, class_names
    
    def train_and_evaluate(self, epochs=100):
        """Train and evaluate the model"""
        X_train, X_test, y_train, y_test, class_names = self.prepare_data()
        
        if X_train is None:
            return
        
        print("\nTraining model...")
        history = self.detector.train_model(
            X_train, y_train, X_test, y_test, class_names, epochs=epochs
        )
        
        # Evaluate the model
        test_loss, test_accuracy = self.detector.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Save the model and scaler
        model_path = "sign_language_model.h5"
        class_names_path = "class_names.pkl"
        scaler_path = "scaler.pkl"
        
        self.detector.save_model(model_path, class_names_path)
        
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\nModel saved as: {model_path}")
        print(f"Class names saved as: {class_names_path}")
        print(f"Scaler saved as: {scaler_path}")
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def test_model_predictions(self, model_path, class_names_path, scaler_path):
        """Test the trained model with some predictions"""
        import pickle
        
        # Load the trained model and scaler
        self.detector.load_model(model_path, class_names_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load test data
        X_train, X_test, y_train, y_test, class_names = self.prepare_data()
        
        if X_test is None:
            print("No test data available")
            return
        
        # Make predictions on test set
        predictions = self.detector.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Show some example predictions
        print("\nExample Predictions:")
        for i in range(min(10, len(X_test))):
            actual = class_names[y_test[i]]
            predicted = class_names[predicted_classes[i]]
            confidence = predictions[i][predicted_classes[i]]
            print(f"Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.3f}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    print("Sign Language Model Trainer")
    print("1. Train new model")
    print("2. Test existing model")
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == '1':
        epochs = int(input("Enter number of epochs (default 100): ") or 100)
        trainer.train_and_evaluate(epochs)
    
    elif choice == '2':
        model_path = input("Enter model path (default: sign_language_model.h5): ") or "sign_language_model.h5"
        class_names_path = input("Enter class names path (default: class_names.pkl): ") or "class_names.pkl"
        scaler_path = input("Enter scaler path (default: scaler.pkl): ") or "scaler.pkl"
        trainer.test_model_predictions(model_path, class_names_path, scaler_path)