"""
Sign Language Detection Application
Main interface for the sign language detection system
"""

import os
import sys
from data_collector import DataCollector
from train_model import ModelTrainer
from video_processor import VideoProcessor

class SignLanguageApp:
    def __init__(self):
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.processor = VideoProcessor()
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("  SIGN LANGUAGE DETECTION SYSTEM")
        print("="*50)
        print("1. Data Collection")
        print("   - Collect training data from webcam or video")
        print("2. Model Training")
        print("   - Train the neural network model")
        print("3. Video Processing")
        print("   - Detect signs in video files")
        print("4. Real-time Detection")
        print("   - Live webcam sign detection")
        print("5. System Information")
        print("6. Exit")
        print("="*50)
    
    def data_collection_menu(self):
        """Handle data collection options"""
        print("\n--- DATA COLLECTION ---")
        print("1. Collect from webcam")
        print("2. Collect from video file")
        print("3. View existing dataset")
        print("4. Back to main menu")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            self.collect_from_webcam()
        elif choice == '2':
            self.collect_from_video()
        elif choice == '3':
            self.view_dataset()
        elif choice == '4':
            return
        else:
            print("Invalid choice!")
    
    def collect_from_webcam(self):
        """Collect data from webcam"""
        sign_name = input("Enter the sign name: ").strip()
        if not sign_name:
            print("Sign name cannot be empty!")
            return
        
        try:
            num_samples = int(input("Number of samples to collect (default 100): ") or 100)
        except ValueError:
            num_samples = 100
        
        print(f"\nStarting data collection for '{sign_name}'")
        print("Instructions:")
        print("- Position your hand clearly in front of the camera")
        print("- Press 's' to start recording samples")
        print("- Keep your hand steady while recording")
        print("- Press 'q' to quit early")
        
        input("Press Enter to continue...")
        
        samples = self.collector.collect_data_from_webcam(sign_name, num_samples)
        
        if samples:
            print(f"Successfully collected {len(samples)} samples for '{sign_name}'")
        else:
            print("No samples collected!")
    
    def collect_from_video(self):
        """Collect data from video file"""
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print("Video file not found!")
            return
        
        sign_name = input("Enter the sign name for this video: ").strip()
        if not sign_name:
            print("Sign name cannot be empty!")
            return
        
        samples = self.collector.collect_data_from_video(video_path, sign_name)
        
        if samples:
            print(f"Successfully extracted {len(samples)} samples from video")
        else:
            print("No samples extracted from video!")
    
    def view_dataset(self):
        """View information about the collected dataset"""
        try:
            X, y, class_names = self.collector.load_dataset()
            
            if len(X) == 0:
                print("No dataset found! Please collect some data first.")
                return
            
            print(f"\nDataset Summary:")
            print(f"Total samples: {len(X)}")
            print(f"Number of classes: {len(class_names)}")
            print(f"Feature dimensions: {X.shape[1]}")
            
            print(f"\nClass distribution:")
            from collections import Counter
            class_counts = Counter(y)
            for i, class_name in enumerate(class_names):
                count = class_counts[i]
                percentage = (count / len(X)) * 100
                print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def training_menu(self):
        """Handle model training"""
        print("\n--- MODEL TRAINING ---")
        
        # Check if dataset exists
        try:
            X, y, class_names = self.collector.load_dataset()
            if len(X) == 0:
                print("No training data found!")
                print("Please collect data first using the Data Collection option.")
                return
            
            print(f"Dataset found: {len(X)} samples, {len(class_names)} classes")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        print("Training options:")
        print("1. Train new model")
        print("2. Test existing model")
        print("3. Back to main menu")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            try:
                epochs = int(input("Enter number of training epochs (default 100): ") or 100)
            except ValueError:
                epochs = 100
            
            print(f"\nStarting training with {epochs} epochs...")
            print("This may take several minutes depending on your hardware.")
            
            history = self.trainer.train_and_evaluate(epochs)
            
            if history:
                print("\nTraining completed successfully!")
                print("Model saved as 'sign_language_model.h5'")
            
        elif choice == '2':
            model_path = "sign_language_model.h5"
            class_names_path = "class_names.pkl"
            scaler_path = "scaler.pkl"
            
            if not os.path.exists(model_path):
                print("No trained model found! Please train a model first.")
                return
            
            self.trainer.test_model_predictions(model_path, class_names_path, scaler_path)
        
        elif choice == '3':
            return
        else:
            print("Invalid choice!")
    
    def video_processing_menu(self):
        """Handle video processing"""
        print("\n--- VIDEO PROCESSING ---")
        
        # Check if model exists
        if not os.path.exists("sign_language_model.h5"):
            print("No trained model found!")
            print("Please train a model first using the Model Training option.")
            return
        
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print("Video file not found!")
            return
        
        output_path = input("Save processed video? Enter output path (or press Enter to skip): ").strip()
        if not output_path:
            output_path = None
        
        show_video = input("Show video while processing? (y/n, default y): ").lower() != 'n'
        
        print(f"\nProcessing video: {video_path}")
        print("This may take a while depending on video length...")
        
        result = self.processor.process_single_video(video_path, output_path, show_video)
        
        if result:
            print(f"\n🎉 DETECTION RESULT 🎉")
            print(f"Detected Sign: {result['sign']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Detection Rate: {result['frequency']}/{result['total_predictions']} frames")
            
            print(f"\nAll detected signs:")
            for sign, count in result['all_predictions'].items():
                percentage = (count / result['total_predictions']) * 100
                print(f"  {sign}: {count} times ({percentage:.1f}%)")
        else:
            print("No signs detected in the video.")
    
    def real_time_detection(self):
        """Start real-time detection"""
        print("\n--- REAL-TIME DETECTION ---")
        
        # Check if model exists
        if not os.path.exists("sign_language_model.h5"):
            print("No trained model found!")
            print("Please train a model first using the Model Training option.")
            return
        
        print("Starting real-time sign language detection...")
        print("Instructions:")
        print("- Position your hand clearly in front of the camera")
        print("- The system will show predictions in real-time")
        print("- Press 'q' to quit")
        
        input("Press Enter to start the camera...")
        
        self.processor.real_time_webcam()
    
    def show_system_info(self):
        """Display system information"""
        print("\n--- SYSTEM INFORMATION ---")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        # Check dependencies
        dependencies = ['tensorflow', 'opencv-python', 'mediapipe', 'numpy', 'matplotlib', 'scikit-learn']
        print("\nDependency status:")
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                print(f"  ✅ {dep}: Installed")
            except ImportError:
                print(f"  ❌ {dep}: Not installed")
        
        # Check for data and model files
        print("\nFile status:")
        files_to_check = [
            ("sign_data/", "Training data directory"),
            ("sign_language_model.h5", "Trained model"),
            ("class_names.pkl", "Class names"),
            ("scaler.pkl", "Data scaler")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                print(f"  ✅ {description}: Found")
            else:
                print(f"  ❌ {description}: Not found")
    
    def run(self):
        """Main application loop"""
        print("Welcome to the Sign Language Detection System!")
        print("Please make sure you have installed all dependencies by running:")
        print("pip install -r requirements.txt")
        
        while True:
            try:
                self.display_menu()
                choice = input("Enter your choice (1-6): ")
                
                if choice == '1':
                    self.data_collection_menu()
                elif choice == '2':
                    self.training_menu()
                elif choice == '3':
                    self.video_processing_menu()
                elif choice == '4':
                    self.real_time_detection()
                elif choice == '5':
                    self.show_system_info()
                elif choice == '6':
                    print("Thank you for using the Sign Language Detection System!")
                    break
                else:
                    print("Invalid choice! Please enter a number between 1-6.")
            
            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please try again or check your input.")

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()