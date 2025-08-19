import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle

# Try to import MediaPipe, fallback if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Using fallback hand detection.")

class SignLanguageDetector:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            # Fallback to basic OpenCV detection
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.hand_cascade.empty():
                print("Warning: Could not load cascade classifier")
            else:
                print("Using face detection as hand approximation (fallback mode)")
        
        self.model = None
        self.class_names = []
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                landmarks.append(hand_data)
        
        # Pad or trim to ensure consistent input size
        if len(landmarks) == 0:
            landmarks = [[0.0] * 63]  # 21 landmarks * 3 coordinates
        elif len(landmarks) == 1:
            landmarks.append([0.0] * 63)  # Add empty second hand
        
        return np.array(landmarks[:2]).flatten()  # Max 2 hands, 126 features total
    
    def create_model(self, num_classes):
        """Create CNN model for sign language classification"""
        model = keras.Sequential([
            layers.Input(shape=(126,)),  # 2 hands * 21 landmarks * 3 coordinates
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
        
        landmarks = self.extract_landmarks(frame)
        landmarks = landmarks.reshape(1, -1)
        
        predictions = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if confidence > 0.7:  # Confidence threshold
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
                
                # Draw landmarks and prediction on frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
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
    
    def real_time_detection(self):
        """Real-time sign language detection using webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make prediction
            prediction = self.predict_sign(frame)
            
            # Draw landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Display prediction
            cv2.putText(frame, f'Sign: {prediction}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    
    # For demonstration, let's create a simple example
    # In practice, you would need a dataset to train the model
    print("Sign Language Detector initialized!")
    print("To use this detector, you need to:")
    print("1. Collect training data")
    print("2. Train the model using train_model()")
    print("3. Save the model using save_model()")
    print("4. Load the model and use process_video() or real_time_detection()")