import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Import appropriate detector
try:
    from sign_language_detector import SignLanguageDetector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    from fallback_detector import FallbackSignLanguageDetector as SignLanguageDetector
    MEDIAPIPE_AVAILABLE = False

class VideoProcessor:
    def __init__(self, model_path="sign_language_model.h5", 
                 class_names_path="class_names.pkl", 
                 scaler_path="scaler.pkl"):
        self.detector = SignLanguageDetector()
        self.scaler = None
        self.load_model_and_scaler(model_path, class_names_path, scaler_path)
    
    def load_model_and_scaler(self, model_path, class_names_path, scaler_path):
        """Load the trained model and scaler"""
        try:
            self.detector.load_model(model_path, class_names_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Model and scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first using train_model.py")
    
    def predict_from_landmarks(self, landmarks):
        """Make prediction from landmarks with proper preprocessing"""
        if self.detector.model is None or self.scaler is None:
            return "No model loaded", 0.0
        
        # Normalize landmarks using the same scaler used during training
        landmarks_normalized = self.scaler.transform(landmarks.reshape(1, -1))
        
        # Make prediction
        predictions = self.detector.model.predict(landmarks_normalized, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if confidence > 0.7:  # Confidence threshold
            return self.detector.class_names[predicted_class], confidence
        else:
            return "Unknown", confidence
    
    def process_single_video(self, video_path, output_path=None, show_video=True):
        """Process a single video file and return the detected sign"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        
        # Video writer setup if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        confidences = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame to reduce computation
            if frame_count % 3 == 0:
                landmarks = self.detector.extract_landmarks(frame)
                
                if np.sum(landmarks) != 0:  # Valid landmarks detected
                    prediction, confidence = self.predict_from_landmarks(landmarks)
                    predictions.append(prediction)
                    confidences.append(confidence)
                
                # Draw landmarks on frame
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
                
                # Add prediction text to frame
                if predictions:
                    current_pred = predictions[-1]
                    current_conf = confidences[-1]
                    text = f'Sign: {current_pred} ({current_conf:.2f})'
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                
                # Progress indicator
                progress = (frame_count / total_frames) * 100
                cv2.putText(frame, f'Progress: {progress:.1f}%', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show video if requested
            if show_video:
                cv2.imshow('Sign Language Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")
        
        cap.release()
        if output_path:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Determine final prediction
        if predictions:
            from collections import Counter
            # Get most common prediction
            prediction_counts = Counter(predictions)
            most_common_sign = prediction_counts.most_common(1)[0][0]
            
            # Calculate average confidence for the most common sign
            sign_confidences = [conf for pred, conf in zip(predictions, confidences) 
                              if pred == most_common_sign]
            avg_confidence = np.mean(sign_confidences)
            
            print(f"\nVideo processing complete!")
            print(f"Detected sign: {most_common_sign}")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Detection frequency: {prediction_counts[most_common_sign]}/{len(predictions)} frames")
            
            return {
                'sign': most_common_sign,
                'confidence': avg_confidence,
                'frequency': prediction_counts[most_common_sign],
                'total_predictions': len(predictions),
                'all_predictions': prediction_counts
            }
        else:
            print("No sign detected in the video")
            return None
    
    def batch_process_videos(self, video_paths, output_dir=None):
        """Process multiple videos"""
        results = {}
        
        for i, video_path in enumerate(video_paths):
            print(f"\n--- Processing video {i+1}/{len(video_paths)} ---")
            
            output_path = None
            if output_dir:
                import os
                filename = os.path.basename(video_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_processed{ext}")
            
            result = self.process_single_video(video_path, output_path, show_video=False)
            results[video_path] = result
        
        return results
    
    def real_time_webcam(self):
        """Real-time sign language detection using webcam"""
        print("Starting real-time detection. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        prediction_buffer = []
        buffer_size = 10  # Number of predictions to average
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks and make prediction
            landmarks = self.detector.extract_landmarks(frame)
            
            if np.sum(landmarks) != 0:  # Valid landmarks detected
                prediction, confidence = self.predict_from_landmarks(landmarks)
                prediction_buffer.append((prediction, confidence))
                
                # Keep only recent predictions
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
            
            # Draw landmarks
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
            
            # Display averaged prediction
            if prediction_buffer:
                # Get most common prediction from buffer
                from collections import Counter
                predictions = [pred for pred, conf in prediction_buffer]
                prediction_counts = Counter(predictions)
                most_common = prediction_counts.most_common(1)[0][0]
                
                # Calculate average confidence
                relevant_confidences = [conf for pred, conf in prediction_buffer if pred == most_common]
                avg_confidence = np.mean(relevant_confidences)
                
                # Display prediction
                text = f'Sign: {most_common} ({avg_confidence:.2f})'
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Display buffer status
                cv2.putText(frame, f'Buffer: {len(prediction_buffer)}/{buffer_size}', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Real-time Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoProcessor()
    
    print("Sign Language Video Processor")
    print("1. Process single video")
    print("2. Process multiple videos")
    print("3. Real-time webcam detection")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        video_path = input("Enter video path: ")
        output_path = input("Enter output path (optional, press Enter to skip): ") or None
        show_video = input("Show video while processing? (y/n, default y): ").lower() != 'n'
        
        result = processor.process_single_video(video_path, output_path, show_video)
        if result:
            print(f"\nFinal Result: {result}")
    
    elif choice == '2':
        video_paths = input("Enter video paths (comma-separated): ").split(',')
        video_paths = [path.strip() for path in video_paths]
        output_dir = input("Enter output directory (optional): ") or None
        
        results = processor.batch_process_videos(video_paths, output_dir)
        print("\nBatch processing results:")
        for path, result in results.items():
            print(f"{path}: {result}")
    
    elif choice == '3':
        processor.real_time_webcam()