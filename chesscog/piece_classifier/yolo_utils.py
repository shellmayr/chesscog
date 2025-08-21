"""Utilities for YOLO-based chess piece detection."""

import numpy as np
import chess
import cv2
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment

from chesscog.core.dataset import name_to_piece


def detect_pieces_with_yolo(model, img: np.ndarray, corners: np.ndarray, 
                          confidence_threshold: float = 0.5, 
                          iou_threshold: float = 0.4,
                          use_hungarian: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Detect chess pieces using a YOLO model.
    
    Args:
        model: The loaded YOLO model
        img: Input image (RGB)
        corners: Four corner points of the chess board
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for non-maximum suppression
        use_hungarian: Whether to use Hungarian algorithm for optimal assignment
        
    Returns:
        Tuple of (piece_array, raw_detections) where:
        - piece_array: Array of 64 elements with piece objects or None
        - raw_detections: List of raw detection dictionaries
    """
    
    # Option 1: Run YOLO on warped image (current approach)
    # Get warped board image
    warped_board = warp_board_for_yolo(img, corners)
    
    # Run YOLO inference on warped image
    if hasattr(model, 'model') and model.model is not None:
        # Handle our wrapper classes
        yolo_model = model.model
    else:
        yolo_model = model
    
    # Different YOLO libraries have different interfaces
    try:
        # YOLOv8 ultralytics
        results = yolo_model.predict(warped_board, conf=confidence_threshold, iou=iou_threshold)
        detections = parse_ultralytics_results(results[0])
    except:
        try:
            # YOLOv5 format
            results = yolo_model(warped_board, size=640)
            detections = parse_yolov5_results(results)
        except:
            # Custom model - assume it returns tensor
            results = yolo_model(torch.from_numpy(warped_board).permute(2, 0, 1).float().unsqueeze(0))
            detections = parse_custom_results(results, confidence_threshold)
    
    # Convert detections to board squares with improved mapping
    if use_hungarian:
        # Compute board bounds from detections for Hungarian algorithm
        board_bounds = compute_board_bounds_from_detections(detections, padding_factor=0.05)
        if board_bounds:
            print("DEBUG: Using Hungarian algorithm for warped board assignment")
            # Need to convert warped image detections to the board bounds format
            # For warped image, the board bounds are simply the image dimensions
            warped_board_bounds = {
                'min_x': 0,
                'max_x': warped_board.shape[1],
                'min_y': 0,
                'max_y': warped_board.shape[0],
                'width': warped_board.shape[1],
                'height': warped_board.shape[0]
            }
            pieces_array = assign_pieces_to_squares_hungarian(detections, warped_board_bounds)
        else:
            print("DEBUG: No detections found, falling back to improved mapping")
            pieces_array = detections_to_board_array_improved(detections, warped_board.shape[:2])
    else:
        pieces_array = detections_to_board_array_improved(detections, warped_board.shape[:2])
    
    return pieces_array, detections


def warp_board_for_yolo(img: np.ndarray, corners: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Warp the chess board to a square image suitable for YOLO inference.
    
    Args:
        img: Input image
        corners: Four corner points of the chess board
        size: Target size for the warped image
        
    Returns:
        Warped square image
    """
    # Define target points for a square
    target_corners = np.array([
        [0, 0],
        [size-1, 0], 
        [size-1, size-1],
        [0, size-1]
    ], dtype=np.float32)
    
    # Compute transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    
    # Warp the image
    warped = cv2.warpPerspective(img, transformation_matrix, (size, size))
    
    return warped


def parse_ultralytics_results(results) -> List[Dict]:
    """Parse YOLOv8 ultralytics results."""
    detections = []
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        # Get the original image shape and resized shape to compute scaling
        original_shape = results.orig_shape  # (height, width)
        resized_shape = results.boxes.data.shape[0] if hasattr(results.boxes, 'data') else None
        
        print(f"DEBUG: YOLO original_shape: {original_shape}")
        if hasattr(results, 'names'):
            print(f"DEBUG: YOLO detected classes: {results.names}")
            
        for i in range(len(boxes)):
            detections.append({
                'bbox': boxes[i],  # [x1, y1, x2, y2] - these are in original image coordinates
                'confidence': scores[i],
                'class': classes[i],
                'original_shape': original_shape
            })
    
    return detections


def parse_yolov5_results(results) -> List[Dict]:
    """Parse YOLOv5 results."""
    detections = []
    
    # YOLOv5 returns pandas DataFrame
    df = results.pandas().xyxy[0]
    
    for _, row in df.iterrows():
        detections.append({
            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            'confidence': row['confidence'],
            'class': int(row['class'])
        })
    
    return detections


def parse_custom_results(results: torch.Tensor, confidence_threshold: float) -> List[Dict]:
    """Parse custom model results - assume standard YOLO format."""
    detections = []
    
    # Assume results is [batch, predictions, 6] where 6 = [x, y, w, h, conf, class]
    if len(results.shape) == 3:
        results = results[0]  # Remove batch dimension
    
    # Filter by confidence
    confident_detections = results[results[:, 4] > confidence_threshold]
    
    for detection in confident_detections:
        x_center, y_center, width, height, conf, cls = detection[:6]
        
        # Convert from center format to corner format
        x1 = float(x_center - width / 2)
        y1 = float(y_center - height / 2) 
        x2 = float(x_center + width / 2)
        y2 = float(y_center + height / 2)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class': int(cls)
        })
    
    return detections


def detections_to_board_array_improved(detections: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert YOLO detections to a 64-element array representing the chess board with improved mapping.
    
    Args:
        detections: List of detection dictionaries
        image_shape: Shape of the warped board image (height, width)
        
    Returns:
        Array of 64 elements with piece objects or None
    """
    pieces_array = np.full(64, None, dtype=object)
    height, width = image_shape
    
    # Define class names (should match your YOLO training)
    class_names = [
        "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
    ]
    
    # Create a list to track all candidates for each square
    square_candidates = [[] for _ in range(64)]
    
    for detection in detections:
        # Get piece class
        class_id = detection['class']
        if class_id >= len(class_names):
            continue
            
        piece_name = class_names[class_id]
        piece = name_to_piece(piece_name)
        
        # Get bounding box BOTTOM CENTER (where piece touches the board)
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        # Find the closest square(s) to this detection
        best_squares = find_closest_squares(center_x, center_y, width, height)
        
        for square_idx, distance in best_squares:
            if 0 <= square_idx < 64:
                square_candidates[square_idx].append({
                    'piece': piece,
                    'confidence': detection['confidence'],
                    'distance': distance,
                    'detection': detection
                })
    
    # For each square, pick the best candidate
    for square_idx in range(64):
        candidates = square_candidates[square_idx]
        if not candidates:
            continue
            
        # Sort by confidence * (1 / (1 + distance)) to prefer high confidence and close detections
        candidates.sort(key=lambda x: x['confidence'] * (1 / (1 + x['distance'])), reverse=True)
        
        best_candidate = candidates[0]
        pieces_array[square_idx] = best_candidate['piece']
        # Store confidence for comparison
        setattr(best_candidate['piece'], '_confidence', best_candidate['confidence'])
    
    return pieces_array


def find_closest_squares(center_x: float, center_y: float, img_width: int, img_height: int, 
                        max_squares: int = 3) -> List[Tuple[int, float]]:
    """
    Find the closest chess squares to a given pixel coordinate.
    
    Args:
        center_x, center_y: Pixel coordinates
        img_width, img_height: Image dimensions
        max_squares: Maximum number of closest squares to return
        
    Returns:
        List of (square_index, distance) tuples, sorted by distance
    """
    square_size_x = img_width / 8
    square_size_y = img_height / 8
    
    closest_squares = []
    
    for rank in range(8):
        for file in range(8):
            # Calculate the center of this square
            square_center_x = (file + 0.5) * square_size_x
            square_center_y = (rank + 0.5) * square_size_y
            
            # Calculate distance to detection center
            distance = np.sqrt((center_x - square_center_x)**2 + (center_y - square_center_y)**2)
            
            # Convert to square index (rank 0 = rank 8 in chess notation)
            square_idx = (7 - rank) * 8 + file
            
            closest_squares.append((square_idx, distance))
    
    # Sort by distance and return top candidates
    closest_squares.sort(key=lambda x: x[1])
    return closest_squares[:max_squares]


def detections_to_board_array(detections: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert YOLO detections to a 64-element array representing the chess board.
    
    Args:
        detections: List of detection dictionaries
        image_shape: Shape of the warped board image (height, width)
        
    Returns:
        Array of 64 elements with piece objects or None
    """
    pieces_array = np.full(64, None, dtype=object)
    height, width = image_shape
    
    # Define class names (should match your YOLO training)
    class_names = [
        "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
    ]
    
    for detection in detections:
        # Get piece class
        class_id = detection['class']
        if class_id >= len(class_names):
            continue
            
        piece_name = class_names[class_id]
        piece = name_to_piece(piece_name)
        
        # Get bounding box BOTTOM CENTER (where piece touches the board)
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        # Convert pixel coordinates to square index
        square_idx = pixel_to_square(center_x, center_y, width, height)
        
        if 0 <= square_idx < 64:
            # If multiple detections for same square, keep the one with higher confidence
            if (pieces_array[square_idx] is None or 
                getattr(pieces_array[square_idx], '_confidence', 0) < detection['confidence']):
                pieces_array[square_idx] = piece
                # Store confidence for comparison
                setattr(piece, '_confidence', detection['confidence'])
    
    return pieces_array


def pixel_to_square(x: float, y: float, img_width: int, img_height: int) -> int:
    """
    Convert pixel coordinates to chess square index.
    
    Args:
        x, y: Pixel coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        Square index (0-63) or -1 if outside board
    """
    # Convert to file and rank
    file = int(8 * x / img_width)
    rank = int(8 * y / img_height)
    
    # Clamp to valid range
    file = max(0, min(7, file))
    rank = max(0, min(7, rank))
    
    # Convert to square index (rank 0 = rank 8 in chess notation)
    square_idx = (7 - rank) * 8 + file
    
    return square_idx


def load_yolo_model(model_path: str, model_type: str = "auto"):
    """
    Load a YOLO model from file.
    
    Args:
        model_path: Path to the model file
        model_type: Type of YOLO model ("yolov5", "yolov8", "custom", or "auto")
        
    Returns:
        Loaded YOLO model
    """
    if model_type == "auto":
        # Try to detect model type from path or content
        if "yolov8" in str(model_path).lower():
            model_type = "yolov8"
        elif "yolov5" in str(model_path).lower():
            model_type = "yolov5"
        else:
            model_type = "custom"
    
    if model_type == "yolov8":
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except ImportError:
            raise ImportError("Install ultralytics for YOLOv8 support: pip install ultralytics")
    
    elif model_type == "yolov5":
        try:
            import yolov5
            return yolov5.load(model_path)
        except ImportError:
            try:
                return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            except Exception as e:
                raise ImportError(f"Could not load YOLOv5 model: {e}")
    
    else:  # custom
        return torch.load(model_path, map_location="cpu", weights_only=False)


def detect_pieces_with_yolo_on_original(model, img: np.ndarray, corners: np.ndarray, 
                                       confidence_threshold: float = 0.5, 
                                       iou_threshold: float = 0.4,
                                       use_hungarian: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Alternative approach: Run YOLO on original image, then map detections to board squares.
    
    This approach may be more robust as it avoids distortions from warping.
    
    Args:
        model: The loaded YOLO model
        img: Input image (RGB) - original, unwarped
        corners: Four corner points of the chess board in original image
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for non-maximum suppression
        use_hungarian: Whether to use Hungarian algorithm for optimal assignment
        
    Returns:
        Tuple of (piece_array, raw_detections) where:
        - piece_array: Array of 64 elements with piece objects or None
        - raw_detections: List of raw detection dictionaries
    """
    
    # Debug: Print image info
    print(f"DEBUG: YOLO running on image shape: {img.shape}")
    
    # Run YOLO inference on original image
    if hasattr(model, 'model') and model.model is not None:
        yolo_model = model.model
    else:
        yolo_model = model
    
    # Different YOLO libraries have different interfaces
    try:
        # YOLOv8 ultralytics
        results = yolo_model.predict(img, conf=confidence_threshold, iou=iou_threshold)
        detections = parse_ultralytics_results(results[0])
        print(f"DEBUG: YOLO found {len(detections)} detections")
        if detections:
            print(f"DEBUG: First detection bbox: {detections[0]['bbox']}")
            
            # Debug: Count what YOLO actually found
            class_names = [
                "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
                "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
            ]
            yolo_counts = {}
            for det in detections:
                class_id = det['class']
                if class_id < len(class_names):
                    piece_name = class_names[class_id]
                    yolo_counts[piece_name] = yolo_counts.get(piece_name, 0) + 1
            print(f"DEBUG: YOLO detection breakdown: {yolo_counts}")
    except:
        try:
            # YOLOv5 format
            results = yolo_model(img, size=640)
            detections = parse_yolov5_results(results)
        except:
            # Custom model - assume it returns tensor
            results = yolo_model(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0))
            detections = parse_custom_results(results, confidence_threshold)
    
    # Try to infer board boundaries from YOLO detections if corner detection seems wrong
    pieces_array = map_detections_to_board_auto(detections, corners, img.shape[:2], use_hungarian)
    
    return pieces_array, detections


def map_detections_to_board(detections: List[Dict], corners: np.ndarray, 
                          image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Map YOLO detections from original image coordinates to chess board squares.
    
    Args:
        detections: List of detection dictionaries with bbox in original image coords
        corners: Four corner points of the chess board in original image (top-left, top-right, bottom-right, bottom-left)
        image_shape: Shape of original image (height, width)
        
    Returns:
        Array of 64 elements with piece objects or None
    """
    pieces_array = np.full(64, None, dtype=object)
    
    # Define class names (should match your YOLO training)
    class_names = [
        "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
    ]
    
    # Debug: Print corner information (old approach - for reference)
    print(f"DEBUG: [OLD] Corner-based mapping would use corners: {corners}")
    print(f"DEBUG: [OLD] But corner region seems too small compared to YOLO detections")
    
    # Corners are in order: [top-left, top-right, bottom-right, bottom-left]
    # We want to map to a standard board coordinate system where:
    # - (0, 0) is top-left of board
    # - (8, 8) is bottom-right of board  
    board_corners = np.array([
        [0, 0],  # top-left
        [8, 0],  # top-right  
        [8, 8],  # bottom-right
        [0, 8]   # bottom-left
    ], dtype=np.float32)
    
    # Create transformation matrix from original image to board coordinates
    transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), board_corners)
    
    # Create a list to track all candidates for each square
    square_candidates = [[] for _ in range(64)]
    
    for detection in detections:
        # Get piece class
        class_id = detection['class']
        if class_id >= len(class_names):
            continue
            
        piece_name = class_names[class_id]
        piece = name_to_piece(piece_name)
        
        # Get bounding box BOTTOM CENTER (where piece touches the board)
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        # Transform to board coordinates (0-8 range)
        original_point = np.array([[[center_x, center_y]]], dtype=np.float32)
        board_point = cv2.perspectiveTransform(original_point, transform_matrix)[0][0]
        
        board_x = board_point[0]
        board_y = board_point[1]
        
        # Skip detections outside the board
        if board_x < 0 or board_x > 8 or board_y < 0 or board_y > 8:
            continue
            
        # Find closest squares
        best_squares = find_closest_squares_normalized(board_x, board_y)
        
        for square_idx, distance in best_squares:
            if 0 <= square_idx < 64:
                square_candidates[square_idx].append({
                    'piece': piece,
                    'confidence': detection['confidence'],
                    'distance': distance,
                    'detection': detection
                })
    
    # For each square, pick the best candidate
    for square_idx in range(64):
        candidates = square_candidates[square_idx]
        if not candidates:
            continue
            
        # Sort by confidence * (1 / (1 + distance)) to prefer high confidence and close detections
        candidates.sort(key=lambda x: x['confidence'] * (1 / (1 + x['distance'])), reverse=True)
        
        best_candidate = candidates[0]
        pieces_array[square_idx] = best_candidate['piece']
        # Store confidence for comparison
        setattr(best_candidate['piece'], '_confidence', best_candidate['confidence'])
    
    print(f"DEBUG: [OLD] Corner-based mapping - using auto-detection instead")
    return pieces_array


def map_detections_to_board_auto(detections: List[Dict], corners: np.ndarray, 
                               image_shape: Tuple[int, int], 
                               use_hungarian: bool = True) -> np.ndarray:
    """
    Auto-detect board boundaries from YOLO detections and map pieces to squares.
    
    This bypasses corner detection issues by using the piece locations to infer the board.
    
    Args:
        detections: List of detection dictionaries with bbox in original image coords
        corners: Four corner points (may be ignored if unreliable)
        image_shape: Shape of original image (height, width)
        use_hungarian: Whether to use Hungarian algorithm for optimal assignment
        
    Returns:
        Array of 64 elements with piece objects or None
    """
    pieces_array = np.full(64, None, dtype=object)
    
    if not detections:
        print("DEBUG: No detections to map")
        return pieces_array
    
    # Define class names
    class_names = [
        "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
    ]
    
    # Extract all detection centers
    detection_centers = []
    valid_detections = []
    
    for detection in detections:
        class_id = detection['class']
        if class_id >= len(class_names):
            continue
            
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        detection_centers.append([center_x, center_y])
        valid_detections.append(detection)
    
    if len(detection_centers) < 4:
        print(f"DEBUG: Too few detections ({len(detection_centers)}) to infer board")
        return pieces_array
    
    detection_centers = np.array(detection_centers)
    
    # Find bounding box of all detections
    min_x, min_y = detection_centers.min(axis=0)
    max_x, max_y = detection_centers.max(axis=0)
    
    # Add more conservative padding (5% instead of 10%)
    width = max_x - min_x
    height = max_y - min_y
    padding_x = width * 0.05  # Reduced padding
    padding_y = height * 0.05
    
    board_min_x = min_x - padding_x
    board_max_x = max_x + padding_x
    board_min_y = min_y - padding_y
    board_max_y = max_y + padding_y
    
    board_width = board_max_x - board_min_x
    board_height = board_max_y - board_min_y
    
    print(f"DEBUG: Auto-detected board region:")
    print(f"  X: {board_min_x:.0f} to {board_max_x:.0f} (width: {board_width:.0f})")
    print(f"  Y: {board_min_y:.0f} to {board_max_y:.0f} (height: {board_height:.0f})")
    print(f"  Based on {len(detection_centers)} detections")
    
    # Debug: Print what YOLO actually detected
    piece_counts = {}
    for detection in valid_detections:
        class_id = detection['class']
        piece_name = class_names[class_id]
        piece_counts[piece_name] = piece_counts.get(piece_name, 0) + 1
    print(f"DEBUG: YOLO detections breakdown: {piece_counts}")
    
    # Create board bounds dictionary for Hungarian algorithm
    board_bounds = {
        'min_x': board_min_x,
        'max_x': board_max_x,
        'min_y': board_min_y,
        'max_y': board_max_y,
        'width': board_width,
        'height': board_height
    }
    
    # Choose assignment method
    if use_hungarian:
        print("DEBUG: Using Hungarian algorithm for piece assignment")
        pieces_array = assign_pieces_to_squares_hungarian(valid_detections, board_bounds)
        return pieces_array
    else:
        print("DEBUG: Using greedy assignment method")
    
    # Create a list to track all candidates for each square
    square_candidates = [[] for _ in range(64)]
    
    # Map each detection to board squares
    for detection in valid_detections:
        class_id = detection['class']
        piece_name = class_names[class_id]
        piece = name_to_piece(piece_name)
        
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        # Convert to board coordinates (0-8 range)
        board_x = 8.0 * (center_x - board_min_x) / board_width
        board_y = 8.0 * (center_y - board_min_y) / board_height
        
        # Debug first few coordinate mappings
        if len(square_candidates[0]) < 5:  # Only print first 5
            print(f"DEBUG: {piece_name} at ({center_x:.0f}, {center_y:.0f}) → board ({board_x:.2f}, {board_y:.2f})")
        
        # Skip detections outside reasonable bounds
        if board_x < -0.5 or board_x > 8.5 or board_y < -0.5 or board_y > 8.5:
            if len(square_candidates[0]) < 5:  # Only print first few rejections
                print(f"DEBUG: REJECTED {piece_name} - out of bounds at board ({board_x:.2f}, {board_y:.2f})")
            continue
        
        # Find closest squares
        best_squares = find_closest_squares_normalized(board_x, board_y)
        
        for square_idx, distance in best_squares:
            if 0 <= square_idx < 64:
                square_candidates[square_idx].append({
                    'piece': piece,
                    'confidence': detection['confidence'],
                    'distance': distance,
                    'detection': detection
                })
    
    # Create a greedy assignment - each detection can only be used once
    used_detections = set()
    assigned_squares = {}
    
    # Create all square-detection pairs with scores
    all_assignments = []
    for square_idx in range(64):
        for candidate in square_candidates[square_idx]:
            score = candidate['confidence'] * (1 / (1 + candidate['distance']))
            all_assignments.append({
                'square_idx': square_idx,
                'detection_id': id(candidate['detection']),
                'candidate': candidate,
                'score': score
            })
    
    # Sort all assignments by score (best first)
    all_assignments.sort(key=lambda x: x['score'], reverse=True)
    
    # Greedy assignment: take the best assignment that doesn't conflict
    num_assigned = 0
    for assignment in all_assignments:
        square_idx = assignment['square_idx']
        detection_id = assignment['detection_id']
        candidate = assignment['candidate']
        
        # Skip if square is already occupied or detection already used
        if square_idx in assigned_squares or detection_id in used_detections:
            continue
        
        # Assign this piece to this square
        pieces_array[square_idx] = candidate['piece']
        setattr(candidate['piece'], '_confidence', candidate['confidence'])
        assigned_squares[square_idx] = True
        used_detections.add(detection_id)
        num_assigned += 1
    
    print(f"DEBUG: Auto-mapping assigned {num_assigned} pieces to board squares (greedy assignment)")
    print(f"DEBUG: Used {len(used_detections)} out of {len(valid_detections)} detections")
    print(f"DEBUG: Using BOTTOM CENTER of bounding boxes for improved perspective handling")
    
    # Debug: Print first few assignments to verify mapping
    print("DEBUG: First few assignments:")
    for i, piece in enumerate(pieces_array[:16]):  # First 2 rows
        if piece is not None:
            square_name = chess.square_name(i)
            confidence = getattr(piece, '_confidence', 0.0)
            print(f"  {square_name}: {piece.symbol()} (conf: {confidence:.3f})")
    
    # Debug: Print the board layout
    print("DEBUG: Final board layout:")
    for rank in range(8):
        rank_str = ""
        for file in range(8):
            square_idx = (7 - rank) * 8 + file  # Convert to chess square indexing
            piece = pieces_array[square_idx]
            if piece is None:
                rank_str += "."
            else:
                rank_str += piece.symbol()
        print(f"  {rank_str}")
    
    return pieces_array


def find_closest_squares_normalized(board_x: float, board_y: float, 
                                  max_squares: int = 3) -> List[Tuple[int, float]]:
    """
    Find the closest chess squares to a given board coordinate (0-8 range).
    
    Args:
        board_x, board_y: Board coordinates (0-8 range)
        max_squares: Maximum number of closest squares to return
        
    Returns:
        List of (square_index, distance) tuples, sorted by distance
    """
    closest_squares = []
    
    for rank in range(8):
        for file in range(8):
            # Calculate the center of this square (in 0-8 coordinates)
            square_center_x = file + 0.5
            square_center_y = rank + 0.5
            
            # Calculate distance to detection center
            distance = np.sqrt((board_x - square_center_x)**2 + (board_y - square_center_y)**2)
            
            # Convert to square index (rank 0 = rank 8 in chess notation)
            square_idx = (7 - rank) * 8 + file
            
            closest_squares.append((square_idx, distance))
    
    # Sort by distance and return top candidates
    closest_squares.sort(key=lambda x: x[1])
    return closest_squares[:max_squares]


def assign_pieces_to_squares_hungarian(detections: List[Dict], board_bounds: Dict,
                                     distance_threshold_factor: float = 0.7) -> np.ndarray:
    """
    Assign detected pieces to chess squares using the Hungarian algorithm.
    
    This implements optimal assignment by solving the minimum cost assignment problem,
    where cost is the Euclidean distance between piece centers and square centers.
    
    Args:
        detections: List of detection dictionaries with bbox, confidence, class
        board_bounds: Dictionary with board boundaries {'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'}
        distance_threshold_factor: Factor to multiply average square size by for max assignment distance
        
    Returns:
        Array of 64 elements with piece objects or None
    """
    if not detections:
        return np.full(64, None, dtype=object)
    
    # Define class names
    class_names = [
        "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
    ]
    
    # Step 1: Compute piece centers from bounding boxes
    piece_centers = []
    valid_pieces = []
    
    for detection in detections:
        class_id = detection['class']
        if class_id >= len(class_names):
            continue
        
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2  # Horizontal center
        center_y = bbox[3]  # Bottom edge (where piece sits on board)
        
        # Convert to board coordinates (0-8 range)
        board_x = 8.0 * (center_x - board_bounds['min_x']) / board_bounds['width']
        board_y = 8.0 * (center_y - board_bounds['min_y']) / board_bounds['height']
        
        # Skip pieces outside reasonable bounds
        if board_x < -0.5 or board_x > 8.5 or board_y < -0.5 or board_y > 8.5:
            continue
        
        piece_centers.append([board_x, board_y])
        
        piece_name = class_names[class_id]
        piece = name_to_piece(piece_name)
        setattr(piece, '_confidence', detection['confidence'])
        valid_pieces.append(piece)
    
    if not piece_centers:
        print("DEBUG: No valid pieces found for Hungarian assignment")
        return np.full(64, None, dtype=object)
    
    piece_centers = np.array(piece_centers)
    num_pieces = len(piece_centers)
    
    # Step 2: Define square centers (in 0-8 board coordinate system)
    square_centers = []
    square_indices = []
    
    for rank in range(8):
        for file in range(8):
            square_center_x = file + 0.5
            square_center_y = rank + 0.5
            square_centers.append([square_center_x, square_center_y])
            
            # Convert to square index (rank 0 = rank 8 in chess notation)
            square_idx = (7 - rank) * 8 + file
            square_indices.append(square_idx)
    
    square_centers = np.array(square_centers)
    
    # Step 3: Build cost matrix (distances between pieces and squares)
    # Shape: (num_pieces × 64)
    cost_matrix = np.zeros((num_pieces, 64))
    
    for i in range(num_pieces):
        for j in range(64):
            # Euclidean distance
            distance = np.sqrt((piece_centers[i][0] - square_centers[j][0])**2 + 
                             (piece_centers[i][1] - square_centers[j][1])**2)
            cost_matrix[i, j] = distance
    
    # Step 4: Run Hungarian algorithm
    piece_indices, square_indices_assigned = linear_sum_assignment(cost_matrix)
    
    # Step 5: Filter bad assignments based on distance threshold
    average_square_size = 1.0  # In our 0-8 coordinate system, each square is 1x1
    max_distance = average_square_size * distance_threshold_factor
    
    pieces_array = np.full(64, None, dtype=object)
    num_assigned = 0
    num_filtered = 0
    
    for piece_idx, square_idx in zip(piece_indices, square_indices_assigned):
        distance = cost_matrix[piece_idx, square_idx]
        
        if distance <= max_distance:
            chess_square_idx = square_indices[square_idx]
            pieces_array[chess_square_idx] = valid_pieces[piece_idx]
            num_assigned += 1
        else:
            num_filtered += 1
    
    print(f"DEBUG: Hungarian algorithm assigned {num_assigned} pieces, filtered {num_filtered} bad assignments")
    print(f"DEBUG: Distance threshold: {max_distance:.3f} (factor: {distance_threshold_factor})")
    
    # Debug: Print assignment quality statistics
    if num_assigned > 0:
        assigned_distances = []
        for piece_idx, square_idx in zip(piece_indices, square_indices_assigned):
            distance = cost_matrix[piece_idx, square_idx]
            if distance <= max_distance:
                assigned_distances.append(distance)
        
        if assigned_distances:
            print(f"DEBUG: Assignment distances - mean: {np.mean(assigned_distances):.3f}, "
                  f"max: {np.max(assigned_distances):.3f}, std: {np.std(assigned_distances):.3f}")
    
    return pieces_array


def compute_board_bounds_from_detections(detections: List[Dict], padding_factor: float = 0.05) -> Dict:
    """
    Compute board boundaries from YOLO detection positions.
    
    Args:
        detections: List of detection dictionaries
        padding_factor: Amount of padding to add around detection bounds
        
    Returns:
        Dictionary with board bounds information
    """
    if not detections:
        return None
    
    # Extract all detection centers
    detection_centers = []
    
    for detection in detections:
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = bbox[3]  # Bottom edge
        detection_centers.append([center_x, center_y])
    
    detection_centers = np.array(detection_centers)
    
    # Find bounding box of all detections
    min_x, min_y = detection_centers.min(axis=0)
    max_x, max_y = detection_centers.max(axis=0)
    
    # Add padding
    width = max_x - min_x
    height = max_y - min_y
    padding_x = width * padding_factor
    padding_y = height * padding_factor
    
    return {
        'min_x': min_x - padding_x,
        'max_x': max_x + padding_x,
        'min_y': min_y - padding_y,
        'max_y': max_y + padding_y,
        'width': width + 2 * padding_x,
        'height': height + 2 * padding_y
    }


def test_hungarian_assignment():
    """
    Test function to validate Hungarian algorithm assignment with sample data.
    
    This creates a simple scenario with a few pieces and verifies they are assigned
    to the correct nearest squares.
    """
    print("Testing Hungarian algorithm assignment...")
    
    # Create test detections - simulate 3 pieces
    test_detections = [
        {
            'bbox': [100, 50, 150, 100],  # Center at (125, 100), piece at bottom edge
            'confidence': 0.9,
            'class': 0  # black_bishop
        },
        {
            'bbox': [200, 150, 250, 200],  # Center at (225, 200), piece at bottom edge
            'confidence': 0.8,
            'class': 6  # white_bishop
        },
        {
            'bbox': [300, 250, 350, 300],  # Center at (325, 300), piece at bottom edge
            'confidence': 0.95,
            'class': 1  # black_king
        }
    ]
    
    # Create board bounds that make it easy to predict square assignments
    # Let's say the board spans from (50,50) to (450,450) - so 400x400 pixels
    # This means each square is 50x50 pixels
    board_bounds = {
        'min_x': 50,
        'max_x': 450,
        'min_y': 50,
        'max_y': 450,
        'width': 400,
        'height': 400
    }
    
    # Run Hungarian assignment
    result = assign_pieces_to_squares_hungarian(test_detections, board_bounds)
    
    # Print results
    assigned_pieces = 0
    for i, piece in enumerate(result):
        if piece is not None:
            square_name = chess.square_name(i)
            confidence = getattr(piece, '_confidence', 0.0)
            print(f"Square {square_name}: {piece.symbol()} (confidence: {confidence:.3f})")
            assigned_pieces += 1
    
    print(f"Total assigned pieces: {assigned_pieces}")
    
    # Expected assignments based on our test data:
    # Detection 1 at (125, 100) -> board coords (1.5, 1.0) -> should be close to square b8 or similar
    # Detection 2 at (225, 200) -> board coords (3.5, 3.0) -> should be close to square d5 or similar
    # Detection 3 at (325, 300) -> board coords (5.5, 5.0) -> should be close to square f3 or similar
    
    return result


def visualize_assignment_test():
    """
    Create a more comprehensive test that shows the assignment process step by step.
    """
    print("\n" + "="*60)
    print("HUNGARIAN ALGORITHM ASSIGNMENT TEST")
    print("="*60)
    
    # Test with a realistic scenario - opening position pieces
    test_detections = [
        # White pieces (back rank)
        {'bbox': [50, 350, 100, 400], 'confidence': 0.9, 'class': 11},   # white_rook at a1
        {'bbox': [150, 350, 200, 400], 'confidence': 0.85, 'class': 8},  # white_knight at b1
        {'bbox': [250, 350, 300, 400], 'confidence': 0.88, 'class': 6},  # white_bishop at c1
        {'bbox': [350, 350, 400, 400], 'confidence': 0.95, 'class': 10}, # white_queen at d1
        {'bbox': [450, 350, 500, 400], 'confidence': 0.92, 'class': 7},  # white_king at e1
        
        # Black pieces (back rank)
        {'bbox': [50, 50, 100, 100], 'confidence': 0.87, 'class': 5},    # black_rook at a8
        {'bbox': [150, 50, 200, 100], 'confidence': 0.83, 'class': 2},   # black_knight at b8
        {'bbox': [250, 50, 300, 100], 'confidence': 0.86, 'class': 0},   # black_bishop at c8
        {'bbox': [350, 50, 400, 100], 'confidence': 0.94, 'class': 4},   # black_queen at d8
        {'bbox': [450, 50, 500, 100], 'confidence': 0.91, 'class': 1},   # black_king at e8
        
        # Some pawns
        {'bbox': [50, 300, 100, 350], 'confidence': 0.8, 'class': 9},    # white_pawn at a2
        {'bbox': [150, 300, 200, 350], 'confidence': 0.78, 'class': 9},  # white_pawn at b2
        {'bbox': [50, 100, 100, 150], 'confidence': 0.82, 'class': 3},   # black_pawn at a7
        {'bbox': [150, 100, 200, 150], 'confidence': 0.79, 'class': 3},  # black_pawn at b7
    ]
    
    # Board spans full 8x8 grid from (0,0) to (560,560) with some padding
    board_bounds = {
        'min_x': 0,
        'max_x': 560,
        'min_y': 0,  
        'max_y': 560,
        'width': 560,
        'height': 560
    }
    
    print(f"Testing with {len(test_detections)} pieces...")
    print(f"Board bounds: {board_bounds['width']}x{board_bounds['height']} pixels")
    print(f"Each square ≈ {board_bounds['width']//8}x{board_bounds['height']//8} pixels")
    
    # Run Hungarian assignment
    result = assign_pieces_to_squares_hungarian(test_detections, board_bounds, distance_threshold_factor=0.8)
    
    # Print the board
    print("\nFinal board layout (Hungarian assignment):")
    for rank in range(8):
        rank_str = f"{8-rank} "
        for file in range(8):
            square_idx = (7 - rank) * 8 + file
            piece = result[square_idx]
            if piece is None:
                rank_str += " ."
            else:
                rank_str += f" {piece.symbol()}"
        print(rank_str)
    print("   a b c d e f g h")
    
    # Count assignments
    assigned = sum(1 for p in result if p is not None)
    print(f"\nAssigned {assigned} out of {len(test_detections)} detected pieces")
    
    return result
