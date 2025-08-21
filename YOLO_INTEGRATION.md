# YOLO Integration for Chess Piece Detection

This document explains how to use YOLO models for chess piece detection in ChessCog.

## Overview

The ChessCog recognition pipeline now supports YOLO models for direct piece detection on chess boards. Instead of the traditional two-stage approach (occupancy classification + piece classification), YOLO models can detect and classify pieces in a single pass.

## Supported YOLO Models

- **YOLOv5**: Using `ultralytics/yolov5` or the `yolov5` package
- **YOLOv8**: Using the `ultralytics` package  
- **CustomYOLO**: Any PyTorch-based YOLO model

## Setup

### Install Dependencies

For YOLOv8 (recommended):
```bash
pip install ultralytics
```

For YOLOv5:
```bash
pip install yolov5
# OR use PyTorch Hub (no extra installation needed)
```

### Configure Your Model

1. **Update the model path** in the configuration file:
   ```yaml
   # config/piece_classifier/YOLOv8.yaml
   TRAINING:
     MODEL:
       MODEL_PATH: "/path/to/your/model/best.pt"
   ```

2. **Adjust detection parameters** if needed:
   ```yaml
   YOLO:
     CONFIDENCE_THRESHOLD: 0.5  # Minimum confidence for detections
     IOU_THRESHOLD: 0.4         # IoU threshold for NMS
     INPUT_SIZE: [640, 640]     # Model input size
   ```

## Current Configuration

Your YOLO model is currently configured to use:
```
Model: /Users/shellmayr/code/chess-yolo/models/chess_comprehensive_20250820_095632/weights/best.pt
Type: YOLOv8n (based on training args)
Confidence: 0.5
IoU Threshold: 0.4
```

## Usage

### Basic Recognition

```python
from chesscog.recognition.recognition import ChessRecognizer
import cv2
import chess

# Load and prepare image
img = cv2.imread("chess_board.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create recognizer (automatically detects YOLO models)
recognizer = ChessRecognizer()

# Perform recognition
board, corners = recognizer.predict(img, chess.WHITE)

print(f"Detected position: {board.fen()}")
```

### Using the Example Script

```bash
cd /Users/shellmayr/code/chesscog
python examples/yolo_recognition.py path/to/your/chess_image.jpg
```

### Timed Recognition

```python
from chesscog.recognition.recognition import TimedChessRecognizer

recognizer = TimedChessRecognizer()
board, corners, times = recognizer.predict(img)

print(f"Corner detection: {times['corner_detection']:.3f}s")
print(f"Piece detection: {times['piece_classification']:.3f}s")
```

## Model Requirements

Your YOLO model should be trained to detect the following classes:
- black_bishop
- black_king  
- black_knight
- black_pawn
- black_queen
- black_rook
- white_bishop
- white_king
- white_knight
- white_pawn
- white_queen
- white_rook

## Switching Between CNN and YOLO

The system automatically detects whether to use CNN or YOLO based on the model configuration. To switch back to CNN:

1. Use a different `classifiers_folder` when creating the recognizer:
   ```python
   recognizer = ChessRecognizer(classifiers_folder=URI("models://cnn_models"))
   ```

2. Or modify the configuration files to use traditional CNN models.

## Performance Notes

- **YOLO advantages**: Single-stage detection, potentially faster, handles overlapping pieces better
- **YOLO requirements**: Requires the entire board to be visible, model must be trained on chess data
- **CNN advantages**: Works with partial boards, separate occupancy/piece classification stages

## Troubleshooting

### Import Errors
```
ImportError: Install ultralytics for YOLOv8 support
```
**Solution**: `pip install ultralytics`

### Model Loading Errors
```
ValueError: Model not loaded. Call load_model() first.
```
**Solution**: Ensure the `MODEL_PATH` in your config points to a valid model file.

### Detection Issues
- **Low confidence detections**: Reduce `CONFIDENCE_THRESHOLD` 
- **Missing pieces**: Check if your model was trained on similar board perspectives
- **Duplicate detections**: Adjust `IOU_THRESHOLD`

## Advanced Configuration

### Custom YOLO Models

For custom YOLO implementations:

```yaml
# config/piece_classifier/CustomYOLO.yaml
_BASE_: config://piece_classifier/_base_yolo.yaml

TRAINING:
  MODEL:
    REGISTRY: PIECE_CLASSIFIER
    NAME: CustomYOLO
    MODEL_PATH: "/path/to/your/custom_model.pt"
```

### Multiple Model Configurations

Create different config files for different models and switch between them as needed.

## Integration Complete

Your YOLO model from `chess_comprehensive_20250820_095632/weights/best.pt` is now fully integrated and ready to use! The system will automatically use YOLO-based detection when you create a `ChessRecognizer` instance.
