"""
Demo Script for Sign Language Detection App
This script demonstrates the complete workflow of the application
"""

import os
import numpy as np
from sign_language_detector import SignLanguageDetector
from data_collector import DataCollector
from train_model import ModelTrainer
from video_processor import VideoProcessor

def create_demo_data():
    """Create some dummy data for demonstration"""
    print("Creating demo data...")
    
    # Simulate collecting data for 3 signs: Hello, Thank You, Please
    signs = ['Hello', 'Thank_You', 'Please']
    
    collector = DataCollector()
    
    for sign in signs:
        print(f"Creating dummy data for '{sign}'...")
        
        # Create 50 dummy samples per sign
        dummy_samples = []
        for i in range(50):
            # Generate random but somewhat realistic hand landmark data
            # Each sample has 126 features (2 hands * 21 landmarks * 3 coordinates)
            sample = np.random.rand(126) * 0.5 + np.random.rand() * 0.3
            
            # Add some pattern to differentiate signs
            if sign == 'Hello':
                sample[0:20] *= 1.5  # Emphasize certain landmarks
            elif sign == 'Thank_You':
                sample[20:40] *= 1.5
            else:  # Please
                sample[40:60] *= 1.5
            
            dummy_samples.append(sample)
        
        # Save dummy data
        import pickle
        data_path = os.path.join(collector.data_dir, f"{sign}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(dummy_samples, f)
        
        print(f"Created {len(dummy_samples)} samples for {sign}")
    
    print("Demo data created successfully!")
    return signs

def run_demo():
    """Run a complete demonstration of the system"""
    print("=" * 60)
    print("SIGN LANGUAGE DETECTION SYSTEM - DEMO")
    print("=" * 60)
    
    # Step 1: Create demo data
    print("\n1. Creating demo training data...")
    signs = create_demo_data()
    
    # Step 2: Train model
    print("\n2. Training the model...")
    trainer = ModelTrainer()
    
    try:
        history = trainer.train_and_evaluate(epochs=20)  # Quick training for demo
        print("✅ Model training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 3: Test the model
    print("\n3. Testing the trained model...")
    try:
        collector = DataCollector()
        X, y, class_names = collector.load_dataset()
        
        if len(X) > 0:
            # Create a simple test
            detector = SignLanguageDetector()
            detector.load_model("sign_language_model.h5", "class_names.pkl")
            
            # Test with a few samples
            print(f"Model loaded. Classes: {class_names}")
            print("Testing with sample data...")
            
            # Use the scaler
            import pickle
            with open("scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            
            # Test a few samples
            test_samples = X[:5]
            test_samples_scaled = scaler.transform(test_samples)
            predictions = detector.model.predict(test_samples_scaled, verbose=0)
            
            for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
                predicted_class = np.argmax(pred)
                confidence = pred[predicted_class]
                print(f"Sample {i+1}: Predicted '{class_names[predicted_class]}' with {confidence:.3f} confidence")
            
            print("✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
    
    # Step 4: Show system status
    print("\n4. System Status:")
    files_created = [
        "sign_language_model.h5",
        "class_names.pkl", 
        "scaler.pkl",
        "sign_data/"
    ]
    
    for file_path in files_created:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python main.py' to use the full application")
    print("2. Try real-time detection with your webcam")
    print("3. Process your own video files")
    print("4. Collect real data for better accuracy")
    print("\nNote: This demo used synthetic data for illustration.")
    print("For real applications, collect actual sign language data!")

def cleanup_demo():
    """Clean up demo files"""
    print("Cleaning up demo files...")
    
    files_to_remove = [
        "sign_language_model.h5",
        "class_names.pkl",
        "scaler.pkl",
        "training_history.png"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
    
    # Remove demo data directory
    import shutil
    if os.path.exists("sign_data"):
        shutil.rmtree("sign_data")
        print("Removed sign_data directory")
    
    print("Demo cleanup completed!")

if __name__ == "__main__":
    print("Sign Language Detection Demo")
    print("1. Run demo")
    print("2. Cleanup demo files")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        run_demo()
    elif choice == '2':
        cleanup_demo()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")