# TSL (Tanzanian Sign Language) - Bridging Silence

## Overview
This application uses machine learning to recognize Tanzanian Sign Language (TSL) hand gestures in real-time and convert them to text and speech.

## Model Input Data Format

### Input Shape
The MLP model (`mlp_tsl_static.pkl`) expects input data in the following format:
- **Shape**: `(1, 63)` - a 2D numpy array
- **Data Type**: `float32`
- **Value Range**: `0.0` to `1.0` (normalized)

### Feature Composition
The 63 features represent **21 hand landmarks** detected by MediaPipe, each with 3 coordinates:

| Feature Index | Description |
|--------------|-------------|
| 0-2 | Landmark 0 (Wrist): x, y, z |
| 3-5 | Landmark 1 (Thumb CMC): x, y, z |
| 6-8 | Landmark 2 (Thumb MCP): x, y, z |
| 9-11 | Landmark 3 (Thumb IP): x, y, z |
| 12-14 | Landmark 4 (Thumb Tip): x, y, z |
| 15-17 | Landmark 5 (Index MCP): x, y, z |
| 18-20 | Landmark 6 (Index PIP): x, y, z |
| 21-23 | Landmark 7 (Index DIP): x, y, z |
| 24-26 | Landmark 8 (Index Tip): x, y, z |
| ... | ... (pattern continues for middle, ring, and pinky fingers) |
| 60-62 | Landmark 20 (Pinky Tip): x, y, z |

**Total**: 21 landmarks × 3 coordinates = **63 features**

### MediaPipe Hand Landmarks Reference
```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
    |   |___|___|___|
    1       |
    |       0 (Wrist)
    0
```

- **0**: Wrist
- **1-4**: Thumb (CMC, MCP, IP, Tip)
- **5-8**: Index finger (MCP, PIP, DIP, Tip)
- **9-12**: Middle finger (MCP, PIP, DIP, Tip)
- **13-16**: Ring finger (MCP, PIP, DIP, Tip)
- **17-20**: Pinky finger (MCP, PIP, DIP, Tip)

### Data Normalization Process

The raw landmark coordinates are normalized using **min-max scaling**:

```python
def normalize_landmarks(landmarks):
    # Input: list of 21 landmarks, each with (x, y, z)
    coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)  # Shape: (21, 3)
    
    # Find min and max for each dimension
    coords_min = coords.min(axis=0)  # Shape: (3,)
    coords_max = coords.max(axis=0)  # Shape: (3,)
    
    # Apply min-max normalization
    norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
    
    # Flatten to single row
    return norm_coords.flatten().reshape(1, -1)  # Shape: (1, 63)
```

**Normalization Formula**:
```
normalized_value = (value - min_value) / (max_value - min_value + epsilon)
```
where `epsilon = 1e-6` prevents division by zero.

### Example Input Data

```python
# Raw landmarks from MediaPipe (21 landmarks with x, y, z)
landmarks = [
    (0.5, 0.6, 0.1),   # Landmark 0 (Wrist)
    (0.45, 0.55, 0.09), # Landmark 1 (Thumb CMC)
    # ... (19 more landmarks)
]

# After normalization
X = normalize_landmarks(landmarks)
# X.shape = (1, 63)
# X[0] = [0.234, 0.567, 0.123, ..., 0.890]  # 63 normalized values

# Prediction
predicted_index = model.predict(X)[0]
predicted_letter = label_encoder.inverse_transform([predicted_index])[0]
```

## Model Output

- **Output**: Integer index (0-25) representing letters A-Z
- **Label Encoding**: Uses scikit-learn's `LabelEncoder`
  - `0` → 'A'
  - `1` → 'B'
  - ...
  - `25` → 'Z'

## Application Features

### Real-time Recognition
- Detects hand gestures via webcam
- Predicts TSL letters when hand is held steady for 1 second
- Automatically builds words and sentences

### Text-to-Speech
- Converts recognized text to Swahili speech using Azure Cognitive Services
- Voice: `sw-KE-ZuriNeural`

### User Controls
- **Start**: Begin video capture
- **Stop**: Stop video capture
- **Clear**: Reset all text
- **Speak**: Convert current sentence to speech
- **Del Letter**: Remove last letter
- **Del Word**: Remove last word

## Technical Requirements

### Dependencies
```bash
numpy
mediapipe
scikit-learn
joblib
opencv-python
pillow
azure-cognitiveservices-speech
```

### Installation
```bash
pip install numpy mediapipe scikit-learn joblib opencv-python pillow azure-cognitiveservices-speech
```

### Model Files Required
- `mlp_tsl_static.pkl` - Trained MLP classifier

### Azure Configuration
Set your Azure Speech API credentials:
```python
speech_key = "YOUR_AZURE_SPEECH_KEY"
service_region = "YOUR_REGION"  # e.g., "eastus"
```

## Usage

```bash
python app.py
```

1. Click **Start** to begin webcam capture
2. Show TSL hand signs to the camera
3. Hold gesture for 1 second to register a letter
4. Remove hand for 2 seconds to add a space
5. Remove hand for 5 seconds to complete a word
6. Click **Speak** to hear the text in Swahili
7. Use **Del Letter** or **Del Word** to correct mistakes

## Data Collection & Training

To train a similar model, you would need:
1. **Dataset**: Hand landmark coordinates (63 features per sample) with corresponding letter labels
2. **Format**: CSV or similar with 63 columns for features + 1 column for labels
3. **Preprocessing**: Apply the same min-max normalization
4. **Model**: MLP classifier (e.g., scikit-learn's `MLPClassifier`)
5. **Export**: Save using `joblib.dump(model, 'mlp_tsl_static.pkl')`

## License
[Specify your license here]

## Contributors
[Add contributors]

## Acknowledgments
- MediaPipe for hand tracking
- Azure Cognitive Services for text-to-speech
- Tanzanian Sign Language community