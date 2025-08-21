# ChessCog Debug Visualizer

A web application for debugging the chess recognition pipeline step-by-step.

## Setup

1. Install dependencies:
```bash
pip install -r webapp_requirements.txt
```

2. Make sure you have the trained models in your `models://` directory (usually `~/chess_data/models/`)

3. Run the web app:
```bash
python debug_webapp.py
```

4. Open your browser to `http://localhost:5000`

## Usage

1. Upload a chess board image
2. Select whether it's White's turn or Black's turn  
3. Click "Analyze Image"
4. The app will show you:
   - **Step 1**: Image resizing and preprocessing
   - **Step 2**: Corner detection results (with corners marked)
   - **Step 3**: Board warping (for CNN approach) or YOLO setup
   - **Step 4**: Occupancy classification (CNN) or YOLO detection
   - **Step 5**: Piece classification results (CNN only)
   - **Final Result**: Complete board state in FEN notation

## Debugging Tips

- **Corner Detection Issues**: Look for red circles marking detected corners. If they're wrong, the whole pipeline fails.
- **Warping Problems**: The warped board should look like a perfect top-down view. Distortions here affect everything downstream.
- **Occupancy Errors**: Green squares = occupied, red squares = empty. Mistakes here prevent pieces from being classified.
- **Piece Classification**: Look for question marks (?) - these are squares detected as occupied but piece classifier failed.

## YOLO Fallback Mode

When corner detection fails but you're using YOLO for piece detection, the debug webapp will automatically attempt **YOLO Fallback Mode**:

- **Runs YOLO directly** on the original image without corner constraints
- **Detects pieces anywhere** in the image with bounding boxes
- **Shows piece types and confidence** without board position mapping
- **Useful for debugging** camera/lighting issues that prevent corner detection

This helps distinguish between:
- **No chess pieces in image** (YOLO finds nothing)
- **Chess pieces present but board boundaries unclear** (YOLO finds pieces but no corners)

## Common Issues

1. **"Failed to initialize chess recognizer"**: Make sure your models are in the right location
2. **"Corner detection failed"**: The image might be too blurry, poorly lit, or the board isn't clearly visible
   - **With YOLO**: Check the fallback results to see if pieces are still detectable
   - **With CNN**: No fallback available - corner detection must succeed
3. **No pieces detected**: Usually an occupancy classification problem - check step 4
4. **"RANSAC produced no viable results"**: Corner detection algorithm can't find consistent board geometry
   - Try different lighting, camera angle, or image resolution
   - YOLO fallback mode will still show detected pieces

## File Structure

- `debug_webapp.py` - Main Flask application
- `webapp_requirements.txt` - Additional dependencies for the web app
- Models should be in: `~/chess_data/models/occupancy_classifier/` and `~/chess_data/models/piece_classifier/`
