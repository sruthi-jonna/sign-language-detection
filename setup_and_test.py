"""
Setup and Test Script for Sign Language Detection App
This script tests the installation and creates a simple demo
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"[OK] {pip_name}")
        except ImportError:
            print(f"[MISSING] {pip_name} - Not installed")
            missing_packages.append(pip_name)
    
    # Check MediaPipe separately
    try:
        import mediapipe
        print("[OK] mediapipe")
        print("   MediaPipe is available - full hand tracking enabled")
    except ImportError:
        print("[WARNING] mediapipe - Not available (will use fallback)")
        print("   Basic OpenCV detection will be used instead")
    
    return missing_packages

def test_camera_access():
    """Test if camera is accessible"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("[OK] Camera access successful")
                print(f"   Frame shape: {frame.shape}")
            else:
                print("[WARNING] Camera opened but no frame received")
            cap.release()
        else:
            print("[ERROR] Cannot access camera")
            
    except Exception as e:
        print(f"[ERROR] Camera test failed: {e}")

def create_simple_demo():
    """Create a simple demo to test the system"""
    print("\nCreating simple demo...")
    
    try:
        # Test basic imports
        import numpy as np
        from fallback_detector import FallbackSignLanguageDetector
        
        print("[OK] Basic imports successful")
        
        # Create detector instance
        detector = FallbackSignLanguageDetector()
        print("[OK] Detector created successfully")
        
        # Test feature extraction with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = detector.extract_landmarks(dummy_frame)
        print(f"[OK] Feature extraction working - {len(features)} features extracted")
        
        # Test model creation
        model = detector.create_model(num_classes=3)
        print("[OK] Model creation successful")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Demo creation failed: {e}")
        return False

def show_usage_instructions():
    """Show how to use the application"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. Run the main application:")
    print("   python main.py")
    
    print("\n2. Or try the demo script:")
    print("   python demo_script.py")
    
    print("\n3. Or test individual components:")
    print("   python fallback_detector.py  # Test basic detection")
    print("   python data_collector.py     # Collect training data")
    print("   python train_model.py        # Train the model")
    
    print("\nIMPORTANT NOTES:")
    print("- MediaPipe is not available for Python 3.13")
    print("- The app will use basic OpenCV detection instead")
    print("- For better accuracy, consider using Python 3.9-3.11")
    print("- You can still collect data and train models")
    
    print("\nWorkflow:")
    print("1. Collect training data for different signs")
    print("2. Train the neural network model") 
    print("3. Test with video files or real-time detection")

if __name__ == "__main__":
    print("Sign Language Detection - Setup and Test")
    print("="*50)
    
    # Check Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    if sys.version_info >= (3, 12):
        print("WARNING: Python 3.12+ detected - MediaPipe may not be available")
        print("   Using fallback detection method")
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        sys.exit(1)
    
    # Test camera
    test_camera_access()
    
    # Create demo
    demo_success = create_simple_demo()
    
    if demo_success:
        show_usage_instructions()
    else:
        print("\n[ERROR] Setup incomplete. Please check the errors above.")