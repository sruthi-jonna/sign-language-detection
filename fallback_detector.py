"""
Fallback Sign Language Detector
Uses OpenCV for basic hand detection when MediaPipe is not available
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle

class FallbackSignLanguageDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
        
        # Initialize OpenCV cascade classifier for detection
        # Note: This is a basic fallback - MediaPipe provides much better hand tracking
        # Using face detection as a simple fallback since hand cascades aren't readily available
        self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.hand_cascade.empty():
            print("Warning: Could not load cascade classifier")
        else:
            print("Fallback detector initialized (using face detection as hand approximation)")
    
    def extract_features_from_roi(self, roi):
        """Extract simple features from hand region of interest"""
        if roi is None or roi.size == 0:
            return np.zeros(64)  # Return zero features if no ROI
        
        # Resize ROI to standard size
        roi_resized = cv2.resize(roi, (32, 32))
        
        # Convert to grayscale if needed
        if len(roi_resized.shape) == 3:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_resized
        
        # Extract simple statistical features
        features = []
        
        # Basic statistics
        features.append(np.mean(roi_gray))
        features.append(np.std(roi_gray))
        features.append(np.min(roi_gray))
        features.append(np.max(roi_gray))
        
        # Histogram features (simplified)
        hist = cv2.calcHist([roi_gray], [0], None, [8], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(roi_gray, 50, 150)
        features.append(np.sum(edges > 0))
        
        # Corner features
        corners = cv2.cornerHarris(roi_gray.astype(np.float32), 2, 3, 0.04)
        features.append(np.sum(corners > 0.01 * corners.max()))
        
        # HOG-like features (simplified)
        dx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Divide into grid and compute mean magnitudes
        h, w = magnitude.shape
        grid_h, grid_w = h // 4, w // 4
        for i in range(4):
            for j in range(4):
                grid_region = magnitude[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                features.append(np.mean(grid_region))
        
        # Pad or trim to exactly 64 features
        features = np.array(features)
        if len(features) < 64:
            features = np.pad(features, (0, 64 - len(features)))
        else:
            features = features[:64]
        
        return features
    
    def extract_landmarks(self, frame):
        """Extract hand features from frame using basic OpenCV detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect hands (or faces as fallback)
        detections = self.hand_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(detections) == 0:
            # No detection found, return zero features
            return np.zeros(128)  # 2 hands * 64 features each
        
        # Extract features from up to 2 detections
        features = []
        for i, (x, y, w, h) in enumerate(detections[:2]):
            roi = frame[y:y+h, x:x+w]
            hand_features = self.extract_features_from_roi(roi)
            features.extend(hand_features)
        
        # Pad to exactly 128 features (2 hands * 64 features)
        if len(features) < 128:
            features.extend([0.0] * (128 - len(features)))
        else:
            features = features[:128]
        
        return np.array(features)
    
    def create_model(self, num_classes):
        """Create a simple neural network for classification"""
        model = keras.Sequential([
            layers.Input(shape=(128,)),  # 128 features from hand detection
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, class_names, epochs=50):
        """Train the model with provided data"""
        self.class_names = class_names
        num_classes = len(class_names)
        
        self.model = self.create_model(num_classes)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path, class_names_path):
        """Save trained model and class names"""
        if self.model:
            self.model.save(model_path)
            with open(class_names_path, 'wb') as f:
                pickle.dump(self.class_names, f)
    
    def load_model(self, model_path, class_names_path):
        """Load trained model and class names"""
        self.model = keras.models.load_model(model_path)
        with open(class_names_path, 'rb') as f:
            self.class_names = pickle.load(f)
    
    def predict_sign(self, frame):
        """Predict sign language from frame"""
        if self.model is None:
            return "No model loaded"
        
        features = self.extract_landmarks(frame)
        features = features.reshape(1, -1)
        
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if confidence > 0.6:  # Lower threshold for basic detection
            return self.class_names[predicted_class]
        else:
            return "Unknown"
    
    def process_video(self, video_path, output_path=None):
        """Process video and detect signs"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to reduce computation
            if frame_count % 5 == 0:
                prediction = self.predict_sign(frame)
                predictions.append(prediction)
                
                # Draw detection rectangles
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.hand_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in detections:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add prediction text
                cv2.putText(frame, f'Prediction: {prediction}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        
        # Return most common prediction
        if predictions:
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0][0]
            return most_common
        return "No prediction"

if __name__ == "__main__":
    print("Fallback Sign Language Detector (without MediaPipe)")
    print("This uses basic OpenCV detection instead of advanced hand tracking.")
    print("For better accuracy, please install MediaPipe when it becomes available for your Python version.")
    
    detector = FallbackSignLanguageDetector()
    
    # Test basic functionality
    print("Testing basic video capture...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera access successful")
        cap.release()
    else:
        print("❌ Camera access failed")
    
    print("Fallback detector ready!")