# TSL Model - Data Format Specification

## Overview
This document describes the exact data format used by the TSL (Tanzanian Sign Language) recognition model, including input structure, coordinate systems, and normalization specifications.

## Raw Input Data Structure

### JSON Format
```json
{
  "landmarks": [
    [0.338775, 0.707677, 0.000000],
    [0.359596, 0.690019, -0.064400],
    [0.402614, 0.697840, -0.079851],
    [0.442001, 0.742977, -0.082639],
    [0.478297, 0.772598, -0.079987],
    [0.481200, 0.624843, -0.038444],
    [0.540103, 0.598855, -0.051450],
    [0.579200, 0.580494, -0.059732],
    [0.611820, 0.569921, -0.064929],
    [0.484388, 0.664041, -0.011230],
    [0.522004, 0.718519, -0.034067],
    [0.490152, 0.736844, -0.042311],
    [0.466428, 0.731174, -0.039971],
    [0.479003, 0.698777, 0.009253],
    [0.507311, 0.745670, -0.019042],
    [0.473320, 0.753283, -0.027947],
    [0.456450, 0.737558, -0.024299],
    [0.467371, 0.723029, 0.025279],
    [0.494282, 0.756875, 0.004044],
    [0.473828, 0.764333, -0.002101],
    [0.456327, 0.749471, 0.000011]
  ]
}
```

## Data Specifications

### Array Structure
- **Total landmarks**: 21
- **Coordinates per landmark**: 3 (x, y, z)
- **Total features**: 63
- **Data type**: Float (6 decimal precision in raw data)

### Landmark Mapping

| Index | Landmark Name | Coordinates | Description |
|-------|--------------|-------------|-------------|
| 0 | WRIST | [0.338775, 0.707677, 0.000000] | Base of the hand |
| 1 | THUMB_CMC | [0.359596, 0.690019, -0.064400] | Thumb carpometacarpal joint |
| 2 | THUMB_MCP | [0.402614, 0.697840, -0.079851] | Thumb metacarpophalangeal joint |
| 3 | THUMB_IP | [0.442001, 0.742977, -0.082639] | Thumb interphalangeal joint |
| 4 | THUMB_TIP | [0.478297, 0.772598, -0.079987] | Thumb tip |
| 5 | INDEX_FINGER_MCP | [0.481200, 0.624843, -0.038444] | Index finger metacarpophalangeal joint |
| 6 | INDEX_FINGER_PIP | [0.540103, 0.598855, -0.051450] | Index finger proximal interphalangeal joint |
| 7 | INDEX_FINGER_DIP | [0.579200, 0.580494, -0.059732] | Index finger distal interphalangeal joint |
| 8 | INDEX_FINGER_TIP | [0.611820, 0.569921, -0.064929] | Index finger tip |
| 9 | MIDDLE_FINGER_MCP | [0.484388, 0.664041, -0.011230] | Middle finger metacarpophalangeal joint |
| 10 | MIDDLE_FINGER_PIP | [0.522004, 0.718519, -0.034067] | Middle finger proximal interphalangeal joint |
| 11 | MIDDLE_FINGER_DIP | [0.490152, 0.736844, -0.042311] | Middle finger distal interphalangeal joint |
| 12 | MIDDLE_FINGER_TIP | [0.466428, 0.731174, -0.039971] | Middle finger tip |
| 13 | RING_FINGER_MCP | [0.479003, 0.698777, 0.009253] | Ring finger metacarpophalangeal joint |
| 14 | RING_FINGER_PIP | [0.507311, 0.745670, -0.019042] | Ring finger proximal interphalangeal joint |
| 15 | RING_FINGER_DIP | [0.473320, 0.753283, -0.027947] | Ring finger distal interphalangeal joint |
| 16 | RING_FINGER_TIP | [0.456450, 0.737558, -0.024299] | Ring finger tip |
| 17 | PINKY_MCP | [0.467371, 0.723029, 0.025279] | Pinky metacarpophalangeal joint |
| 18 | PINKY_PIP | [0.494282, 0.756875, 0.004044] | Pinky proximal interphalangeal joint |
| 19 | PINKY_DIP | [0.473828, 0.764333, -0.002101] | Pinky distal interphalangeal joint |
| 20 | PINKY_TIP | [0.456327, 0.749471, 0.000011] | Pinky tip |

## Coordinate System

### X-Axis (Horizontal)
- **Range**: 0.0 to 1.0 (normalized by image width)
- **Direction**: Left (0.0) → Right (1.0)
- **Example**: `0.338775` = 33.88% from left edge

### Y-Axis (Vertical)
- **Range**: 0.0 to 1.0 (normalized by image height)
- **Direction**: Top (0.0) → Bottom (1.0)
- **Example**: `0.707677` = 70.77% from top edge

### Z-Axis (Depth)
- **Range**: Typically -0.1 to 0.1 (relative to wrist)
- **Direction**: Negative (away from camera) → Positive (towards camera)
- **Reference Point**: Wrist (landmark 0) is usually at z ≈ 0.0
- **Example**: `-0.064400` = 6.44cm away from wrist depth

## Data Preprocessing Pipeline

### Step 1: Raw Data Collection
```python
# MediaPipe returns landmarks in this format
landmarks = [
    (0.338775, 0.707677, 0.000000),  # Landmark 0
    (0.359596, 0.690019, -0.064400), # Landmark 1
    # ... (19 more landmarks)
]
```

### Step 2: Convert to NumPy Array
```python
import numpy as np

coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
# Shape: (21, 3)
# Example:
# [[0.338775, 0.707677, 0.000000],
#  [0.359596, 0.690019, -0.064400],
#  ...]
```

### Step 3: Apply Min-Max Normalization
```python
# Calculate min and max for each axis
coords_min = coords.min(axis=0)  # [min_x, min_y, min_z]
coords_max = coords.max(axis=0)  # [max_x, max_y, max_z]

# Normalize each coordinate
norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
# Shape: (21, 3)
# Values now range from 0.0 to 1.0
```

### Step 4: Flatten for Model Input
```python
X = norm_coords.flatten().reshape(1, -1)
# Shape: (1, 63)
# Ready for model.predict(X)
```

## Example: Complete Data Transformation

### Input (Raw)
```python
raw_landmarks = [
    [0.338775, 0.707677, 0.000000],  # Wrist
    [0.359596, 0.690019, -0.064400],  # Thumb CMC
    # ... (19 more)
]
```

### After Reshaping
```python
# Shape: (21, 3)
[[0.338775, 0.707677, 0.000000],
 [0.359596, 0.690019, -0.064400],
 [0.402614, 0.697840, -0.079851],
 ...]
```

### After Normalization
```python
# Example normalized values (actual values depend on min/max)
# Shape: (21, 3)
[[0.000000, 0.523456, 0.891234],
 [0.076234, 0.401234, 0.234567],
 [0.234567, 0.456789, 0.123456],
 ...]
```

### Final Model Input
```python
# Shape: (1, 63)
[[0.000000, 0.523456, 0.891234, 0.076234, 0.401234, ..., 0.567890]]
```

## Value Ranges Summary

| Component | Raw Range | Normalized Range | Notes |
|-----------|-----------|------------------|-------|
| X coordinates | 0.0 - 1.0 | 0.0 - 1.0 | Image-relative horizontal position |
| Y coordinates | 0.0 - 1.0 | 0.0 - 1.0 | Image-relative vertical position |
| Z coordinates | ~-0.1 - 0.1 | 0.0 - 1.0 | Depth relative to wrist, normalized |
| Model input | N/A | 0.0 - 1.0 | All 63 features normalized |

## Data Validation

### Valid Data Criteria
✅ Exactly 21 landmarks  
✅ Each landmark has exactly 3 coordinates  
✅ X and Y values between 0.0 and 1.0  
✅ Z values typically between -0.2 and 0.2  
✅ No NaN or infinite values  

### Invalid Data Examples
❌ Missing landmarks (< 21)  
❌ Extra coordinates per landmark (≠ 3)  
❌ X or Y values outside [0.0, 1.0]  
❌ Extreme Z values (> 0.5 or < -0.5)  
❌ NaN or null values  

## Python Code Example

```python
import numpy as np
import json

# Load raw data
with open('hand_data.json', 'r') as f:
    data = json.load(f)

landmarks = data['landmarks']

# Validate
assert len(landmarks) == 21, "Must have 21 landmarks"
assert all(len(lm) == 3 for lm in landmarks), "Each landmark must have 3 coordinates"

# Convert to numpy array
coords = np.array(landmarks, dtype=np.float32)  # Shape: (21, 3)

# Normalize
coords_min = coords.min(axis=0)
coords_max = coords.max(axis=0)
norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)

# Prepare for model
X = norm_coords.flatten().reshape(1, -1)  # Shape: (1, 63)

# Predict
prediction = model.predict(X)
letter = label_encoder.inverse_transform(prediction)[0]

print(f"Predicted letter: {letter}")
```

## Dataset Format for Training

### CSV Format
```csv
x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
0.338775,0.707677,0.000000,0.359596,0.690019,-0.064400,...,0.456327,0.749471,0.000011,A
0.425123,0.612345,0.001234,0.445678,0.598765,-0.055555,...,0.567890,0.678901,0.012345,B
...
```

### JSON Format
```json
[
  {
    "landmarks": [[0.338775, 0.707677, 0.000000], ...],
    "label": "A"
  },
  {
    "landmarks": [[0.425123, 0.612345, 0.001234], ...],
    "label": "B"
  }
]
```

## MediaPipe Integration

This data format is generated by **MediaPipe Hands** solution:

```python
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Process frame
results = hands.process(rgb_frame)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract landmarks
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        # landmarks is now in the correct format
```

## References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [Hand Landmark Model](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model)
- Main application: `app.py`
- Model file: `mlp_tsl_static.pkl`

## Version
- **Format Version**: 1.0
- **Last Updated**: 2026-02-01
- **Compatible Model**: mlp_tsl_static.pkl