# Sign Language Detection App

A machine learning application that detects sign language gestures from video input using TensorFlow, Keras, and MediaPipe.

## Features

- **Data Collection**: Collect training data from webcam or video files
- **Model Training**: Train a neural network to recognize sign language gestures
- **Video Processing**: Process video files to detect sign language
- **Real-time Detection**: Live webcam detection of sign language gestures
- **Hand Landmark Detection**: Uses MediaPipe for accurate hand tracking

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Run the main application**:
   ```bash
   python main.py
   ```

2. **Follow the workflow**:
   - Start with **Data Collection** to gather training samples
   - Use **Model Training** to train your classifier
   - Use **Video Processing** or **Real-time Detection** to test your model

## Detailed Usage

### 1. Data Collection

Collect training data for different signs:

```bash
python data_collector.py
```

Options:
- Collect from webcam (interactive)
- Process existing video files
- View dataset statistics

**Tips for good data collection**:
- Ensure good lighting
- Position hand clearly in camera view
- Collect samples from different angles
- Aim for 100+ samples per sign for better accuracy

### 2. Model Training

Train your sign language classifier:

```bash
python train_model.py
```

The training process will:
- Load your collected data
- Split into training/validation sets
- Train a neural network model
- Save the trained model and preprocessing components

### 3. Video Processing

Process video files to detect signs:

```bash
python video_processor.py
```

Features:
- Process single or multiple videos
- Save processed videos with annotations
- Batch processing capabilities
- Confidence scoring and frequency analysis

### 4. Real-time Detection

Use your webcam for live detection:

```bash
python video_processor.py
# Select option 3 for real-time detection
```

## File Structure

```
sign language detection/
├── main.py                    # Main application interface
├── sign_language_detector.py  # Core detection class
├── data_collector.py         # Data collection utilities
├── train_model.py            # Model training script
├── video_processor.py        # Video processing utilities
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── sign_data/               # Training data (created automatically)
├── sign_language_model.h5   # Trained model (created after training)
├── class_names.pkl          # Class labels (created after training)
└── scaler.pkl              # Data scaler (created after training)
```

## Technical Details

### Model Architecture

- **Input**: Hand landmark coordinates (126 features: 2 hands × 21 landmarks × 3 coordinates)
- **Architecture**: Dense neural network with dropout layers
- **Output**: Softmax classification over sign classes
- **Preprocessing**: StandardScaler normalization

### Hand Detection

- Uses MediaPipe Hands solution
- Extracts 21 landmark points per hand
- Supports up to 2 hands simultaneously
- 3D coordinates (x, y, z) for each landmark

### Data Format

- Each sample: 126-dimensional feature vector
- Features: Normalized 3D hand landmark coordinates
- Labels: Integer class indices
- Storage: Pickle format for training data

## Troubleshooting

### Common Issues

1. **"No model found" error**
   - Solution: Train a model first using the Model Training option

2. **"No training data found" error**
   - Solution: Collect data first using the Data Collection option

3. **Camera not working**
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Try different camera index (0, 1, 2, etc.)

4. **Poor detection accuracy**
   - Collect more training data
   - Ensure consistent hand positioning during data collection
   - Train for more epochs
   - Check lighting conditions

### Performance Tips

- **For better accuracy**:
  - Collect 100+ samples per sign
  - Use consistent lighting
  - Maintain steady hand positions
  - Include variations in hand angles

- **For better speed**:
  - Process every nth frame instead of all frames
  - Use smaller video resolutions
  - Close other applications during processing

## Requirements

- Python 3.7+
- TensorFlow 2.15+
- OpenCV 4.9+
- MediaPipe 0.10+
- NumPy, Matplotlib, Scikit-learn

## Example Workflow

1. **Collect data for "Hello"**:
   - Run data collector
   - Record 100 samples of "Hello" gesture

2. **Collect data for "Thank you"**:
   - Record 100 samples of "Thank you" gesture

3. **Train model**:
   - Run training script
   - Model learns to distinguish between the two signs

4. **Test with video**:
   - Record a video showing either gesture
   - Process video to get prediction

## Contributing

Feel free to improve this application by:
- Adding more sophisticated model architectures
- Implementing data augmentation
- Adding support for more complex gestures
- Improving the user interface

## License

This project is open source and available under the MIT License.