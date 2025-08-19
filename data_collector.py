import cv2
import numpy as np
import os
import pickle

# Try to import MediaPipe, use fallback if not available
try:
    import mediapipe as mp
    from sign_language_detector import SignLanguageDetector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    from fallback_detector import FallbackSignLanguageDetector as SignLanguageDetector
    MEDIAPIPE_AVAILABLE = False
    print("Using fallback detector - MediaPipe not available")

class DataCollector:
    def __init__(self):
        self.detector = SignLanguageDetector()
        self.data_dir = "sign_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def collect_data_from_webcam(self, sign_name, num_samples=100):
        """Collect training data from webcam for a specific sign"""
        print(f"Collecting data for sign: {sign_name}")
        print("Press 's' to start recording, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        samples = []
        recording = False
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Display instructions
            text = f"Sign: {sign_name} | Samples: {count}/{num_samples}"
            if not recording:
                text += " | Press 's' to start"
            else:
                text += " | Recording..."
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Extract landmarks and add to dataset if recording
            if recording:
                landmarks = self.detector.extract_landmarks(frame)
                if np.sum(landmarks) != 0:  # Valid landmarks detected
                    samples.append(landmarks)
                    count += 1
            
            # Draw landmarks if MediaPipe is available
            if MEDIAPIPE_AVAILABLE and hasattr(self.detector, 'hands'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.detector.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.detector.mp_hands.HAND_CONNECTIONS)
            else:
                # Fallback: draw detection rectangles
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.detector.hand_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in detections:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                print("Started recording...")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if samples:
            data_path = os.path.join(self.data_dir, f"{sign_name}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"Saved {len(samples)} samples for {sign_name}")
        
        return samples
    
    def collect_data_from_video(self, video_path, sign_name):
        """Collect training data from a video file"""
        print(f"Processing video: {video_path} for sign: {sign_name}")
        
        cap = cv2.VideoCapture(video_path)
        samples = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.detector.extract_landmarks(frame)
            if np.sum(landmarks) != 0:  # Valid landmarks detected
                samples.append(landmarks)
        
        cap.release()
        
        # Save collected data
        if samples:
            data_path = os.path.join(self.data_dir, f"{sign_name}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"Saved {len(samples)} samples for {sign_name}")
        
        return samples
    
    def load_dataset(self):
        """Load all collected data and create training dataset"""
        X = []
        y = []
        class_names = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                sign_name = filename[:-4]  # Remove .pkl extension
                class_names.append(sign_name)
                
                with open(os.path.join(self.data_dir, filename), 'rb') as f:
                    samples = pickle.load(f)
                
                for sample in samples:
                    X.append(sample)
                    y.append(len(class_names) - 1)
        
        return np.array(X), np.array(y), class_names

if __name__ == "__main__":
    collector = DataCollector()
    
    print("Sign Language Data Collector")
    print("1. Collect from webcam")
    print("2. Collect from video file")
    print("3. Load existing dataset")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        sign_name = input("Enter sign name: ")
        num_samples = int(input("Number of samples (default 100): ") or 100)
        collector.collect_data_from_webcam(sign_name, num_samples)
    
    elif choice == '2':
        video_path = input("Enter video path: ")
        sign_name = input("Enter sign name: ")
        collector.collect_data_from_video(video_path, sign_name)
    
    elif choice == '3':
        X, y, class_names = collector.load_dataset()
        print(f"Dataset loaded: {len(X)} samples, {len(class_names)} classes")
        print(f"Classes: {class_names}")