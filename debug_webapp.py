#!/usr/bin/env python3
"""
Debug web app for visualizing chess recognition pipeline step-by-step.
"""

import os
import io
import base64
import json
import numpy as np
import cv2
import chess
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path

# ChessCog imports
from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from chesscog.core import device, sort_corner_points
from chesscog.core.dataset import build_transforms, Datasets, name_to_piece
from chesscog.core.exceptions import ChessboardNotLocatedException
from recap import URI, CfgNode as CN

app = Flask(__name__)

def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

class DebuggingChessRecognizer(ChessRecognizer):
    """Extended ChessRecognizer that captures intermediate results for debugging."""
    
    def predict_with_debug(self, img: np.ndarray, turn: chess.Color = chess.WHITE, 
                          custom_conf_threshold=None, custom_iou_threshold=None, use_original_image=False):
        """Run prediction capturing all intermediate steps."""
        debug_info = {
            'steps': [],
            'error': None,
            'final_board': None,
            'final_corners': None,
            'yolo_raw_detections': []
        }
        
        try:
            with torch.no_grad():
                # Step 1: Image preprocessing and corner detection
                original_img = img.copy()
                img_resized, img_scale = resize_image(self._corner_detection_cfg, img)
                debug_info['steps'].append({
                    'name': '1. Image Resize',
                    'description': f'Resized image (scale: {img_scale:.3f})',
                    'image': self._encode_image(img_resized),
                    'metadata': make_json_serializable({'scale': img_scale, 'shape': img_resized.shape})
                })
                
                # Step 2: Corner detection
                try:
                    corners = find_corners(self._corner_detection_cfg, img_resized)
                    corner_viz = self._visualize_corners(img_resized, corners)
                    debug_info['steps'].append({
                        'name': '2. Corner Detection',
                        'description': f'Found 4 corners',
                        'image': self._encode_image(corner_viz),
                        'metadata': make_json_serializable({'corners': corners})
                    })
                except ChessboardNotLocatedException as e:
                    debug_info['error'] = f"Corner detection failed: {str(e)}"
                    return debug_info
                
                # Step 3: Image warping
                if self._use_yolo:
                    # YOLO approach - just show original for warping step
                    debug_info['steps'].append({
                        'name': '3. YOLO Detection (No Warping)',
                        'description': 'YOLO works directly on original image',
                        'image': self._encode_image(img_resized),
                        'metadata': {'approach': 'yolo'}
                    })
                    
                    # Step 4: YOLO piece detection with raw detections
                    approach_name = "Original Image" if use_original_image else "Warped Image"
                    if use_original_image:
                        # Use original image and scale corners back to original size
                        input_img = original_img
                        input_corners = corners / img_scale  # Scale corners UP to original image coordinates
                        print(f"DEBUG: Using original image {original_img.shape}, scaled corners by 1/{img_scale} = {1/img_scale}")
                        print(f"DEBUG: Scaled corners: {input_corners}")
                    else:
                        # Use resized image and resized corners
                        input_img = img_resized
                        input_corners = corners
                        print(f"DEBUG: Using resized image {img_resized.shape}, corners as-is")
                    
                    pieces, raw_detections = self._classify_pieces_yolo_debug(
                        input_img, turn, input_corners, custom_conf_threshold, custom_iou_threshold, use_original_image)
                    
                    # Store raw detections for analysis
                    debug_info['yolo_raw_detections'] = raw_detections
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_detections = []
                    for detection in raw_detections:
                        det_copy = detection.copy()
                        if 'bbox' in det_copy and hasattr(det_copy['bbox'], 'tolist'):
                            det_copy['bbox'] = det_copy['bbox'].tolist()
                        elif 'bbox' in det_copy and isinstance(det_copy['bbox'], (list, tuple)):
                            det_copy['bbox'] = [float(x) for x in det_copy['bbox']]
                        det_copy['confidence'] = float(det_copy['confidence'])
                        det_copy['class'] = int(det_copy['class'])
                        serializable_detections.append(det_copy)
                    
                    # Create visualizations
                    piece_viz = self._visualize_yolo_results(input_img, input_corners, pieces, raw_detections, approach_name)
                    debug_info['steps'].append({
                        'name': f'4a. YOLO Raw Detections ({approach_name})',
                        'description': f'YOLO found {len(raw_detections)} detections with bounding boxes',
                        'image': self._encode_image(piece_viz),
                        'metadata': {
                            'num_raw_detections': len(raw_detections),
                            'confidence_threshold': float(custom_conf_threshold or getattr(self._pieces_cfg.YOLO, 'CONFIDENCE_THRESHOLD', 0.5)),
                            'iou_threshold': float(custom_iou_threshold or getattr(self._pieces_cfg.YOLO, 'IOU_THRESHOLD', 0.4)),
                            'approach': approach_name,
                            'raw_detections': serializable_detections
                        }
                    })
                    
                    # Add intermediate detection mapping visualization
                    mapping_viz = self._visualize_detection_mapping(pieces, raw_detections)
                    
                    # Create assignment details for metadata
                    assignment_details = []
                    for i, piece in enumerate(pieces):
                        if piece is not None:
                            square_name = chess.square_name(i)
                            confidence = getattr(piece, '_confidence', 0.0)
                            assignment_details.append({
                                'square_name': square_name,
                                'piece_symbol': piece.symbol(),
                                'piece_name': str(piece),
                                'confidence': float(confidence)
                            })
                    
                    debug_info['steps'].append({
                        'name': '4b. Detection → Board Mapping',
                        'description': f'Mapped {len(raw_detections)} detections to {np.count_nonzero(pieces != None)} board squares',
                        'image': self._encode_image(mapping_viz),
                        'metadata': {
                            'num_final_pieces': int(np.count_nonzero(pieces != None)),
                            'mapping_method': 'auto_detection_region',
                            'assignment_details': assignment_details
                        }
                    })
                    
                    # Add warped board with pieces visualization
                    warped_viz = self._visualize_warped_board_with_pieces(input_img, input_corners, pieces)
                    debug_info['steps'].append({
                        'name': '4c. Warped Board with Assigned Pieces',
                        'description': f'Warped board view showing {np.count_nonzero(pieces != None)} assigned pieces',
                        'image': self._encode_image(warped_viz),
                        'metadata': {
                            'num_pieces_shown': int(np.count_nonzero(pieces != None)),
                            'visualization_type': 'warped_board_overlay'
                        }
                    })
                    
                    # Add clean final board visualization
                    clean_board_viz = self._visualize_clean_final_board(pieces)
                    debug_info['steps'].append({
                        'name': '4d. Final Board State',
                        'description': f'Clean chess board visualization with {np.count_nonzero(pieces != None)} pieces',
                        'image': self._encode_image(clean_board_viz),
                        'metadata': {
                            'num_pieces': int(np.count_nonzero(pieces != None)),
                            'visualization_type': 'clean_board'
                        }
                    })
                else:
                    # Traditional CNN approach
                    warped_occupancy = create_occupancy_dataset.warp_chessboard_image(img_resized, corners)
                    debug_info['steps'].append({
                        'name': '3. Board Warping',
                        'description': 'Warped board to standard view',
                        'image': self._encode_image(warped_occupancy),
                        'metadata': {'warped_size': warped_occupancy.shape}
                    })
                    
                    # Step 4: Occupancy classification
                    occupancy = self._classify_occupancy(img_resized, turn, corners)
                    occupancy_viz = self._visualize_occupancy(warped_occupancy, occupancy)
                    debug_info['steps'].append({
                        'name': '4. Occupancy Classification',
                        'description': f'Found {np.sum(occupancy)} occupied squares',
                        'image': self._encode_image(occupancy_viz),
                        'metadata': {'occupied_squares': int(np.sum(occupancy))}
                    })
                    
                    # Step 5: Piece classification
                    pieces = self._classify_pieces(img_resized, turn, corners, occupancy)
                    warped_pieces = create_piece_dataset.warp_chessboard_image(img_resized, corners)
                    piece_viz = self._visualize_pieces(warped_pieces, occupancy, pieces)
                    debug_info['steps'].append({
                        'name': '5. Piece Classification',
                        'description': f'Classified {np.count_nonzero(pieces != None)} pieces',
                        'image': self._encode_image(piece_viz),
                        'metadata': {'num_pieces': int(np.count_nonzero(pieces != None))}
                    })
                
                # Final assembly
                board = chess.Board()
                board.clear_board()
                for square, piece in zip(self._squares, pieces):
                    if piece:
                        board.set_piece_at(square, piece)
                corners = corners / img_scale
                
                debug_info['final_board'] = board.fen()
                debug_info['final_corners'] = corners.tolist() if hasattr(corners, 'tolist') else corners
                debug_info['board_unicode'] = str(board)
                
                return make_json_serializable(debug_info)
                
        except Exception as e:
            debug_info['error'] = f"Pipeline error: {str(e)}"
            return make_json_serializable(debug_info)
    
    def _encode_image(self, img):
        """Convert numpy array to base64 encoded string."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            # RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        _, buffer = cv2.imencode('.png', img_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def _visualize_corners(self, img, corners):
        """Draw corners on the image."""
        viz = img.copy()
        corners = corners.astype(int)
        
        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(viz, tuple(corner), 10, (255, 0, 0), 3)
            cv2.putText(viz, str(i), tuple(corner + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw bounding rectangle
        cv2.polylines(viz, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        
        return viz
    
    def _visualize_occupancy(self, warped_img, occupancy):
        """Visualize occupancy classification on warped board."""
        viz = warped_img.copy()
        square_size = viz.shape[0] // 8
        
        for i, is_occupied in enumerate(occupancy):
            row = 7 - (i // 8)  # Chess board coordinates
            col = i % 8
            
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            if is_occupied:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(viz, 'O', (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(viz, 'E', (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return viz
    
    def _visualize_pieces(self, warped_img, occupancy, pieces):
        """Visualize piece classification on warped board."""
        viz = warped_img.copy()
        square_size = viz.shape[0] // 8
        
        for i, (is_occupied, piece) in enumerate(zip(occupancy, pieces)):
            row = 7 - (i // 8)  # Chess board coordinates
            col = i % 8
            
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            if is_occupied and piece:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
                piece_str = piece.symbol()
                cv2.putText(viz, piece_str, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            elif is_occupied:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(viz, '?', (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            else:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        return viz
    
    def _visualize_detection_mapping(self, pieces, raw_detections):
        """Create a visualization showing how detections map to board squares."""
        # Create a large canvas for the visualization
        canvas_size = 600
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Draw 8x8 grid
        square_size = canvas_size // 8
        
        # Draw grid lines
        for i in range(9):
            # Vertical lines
            cv2.line(canvas, (i * square_size, 0), (i * square_size, canvas_size), (100, 100, 100), 1)
            # Horizontal lines
            cv2.line(canvas, (0, i * square_size), (canvas_size, i * square_size), (100, 100, 100), 1)
        
        # Color squares alternately (light board pattern)
        for rank in range(8):
            for file in range(8):
                if (rank + file) % 2 == 1:  # Dark squares
                    x1 = file * square_size
                    y1 = rank * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (240, 217, 181), -1)
        
        # Draw square labels (a1-h8)
        for rank in range(8):
            for file in range(8):
                square_idx = (7 - rank) * 8 + file  # Convert to chess square indexing
                square_name = chess.square_name(square_idx)
                
                x = file * square_size + 5
                y = rank * square_size + 15
                cv2.putText(canvas, square_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw assigned pieces
        for i, piece in enumerate(pieces):
            if piece is not None:
                # Convert square index to rank/file
                rank = 7 - (i // 8)
                file = i % 8
                
                # Calculate position
                x = file * square_size + square_size // 2
                y = rank * square_size + square_size // 2
                
                # Choose color based on piece color
                piece_color = (50, 50, 200) if piece.symbol().islower() else (200, 50, 50)
                
                # Draw piece symbol
                cv2.putText(canvas, piece.symbol(), (x - 15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, piece_color, 3)
                
                # Draw confidence below
                confidence = getattr(piece, '_confidence', 0.0)
                cv2.putText(canvas, f"{confidence:.2f}", (x - 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, piece_color, 1)
        
        # Add title and summary info
        num_pieces = np.count_nonzero(pieces != None)
        title = f"Board Mapping: {num_pieces} pieces assigned"
        cv2.putText(canvas, title, (10, canvas_size - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        subtitle = f"From {len(raw_detections)} raw detections"
        cv2.putText(canvas, subtitle, (10, canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return canvas
    
    def _visualize_warped_board_with_pieces(self, img, corners, pieces):
        """Create a warped board view with detected pieces overlaid."""
        from chesscog.piece_classifier.yolo_utils import warp_board_for_yolo
        
        # Create warped board
        warped_board = warp_board_for_yolo(img, corners, size=640)
        viz = warped_board.copy()
        
        # Draw 8x8 grid overlay
        square_size = viz.shape[0] // 8
        
        # Draw grid lines
        for i in range(9):
            cv2.line(viz, (i * square_size, 0), (i * square_size, viz.shape[0]), (255, 255, 255), 2)
            cv2.line(viz, (0, i * square_size), (viz.shape[1], i * square_size), (255, 255, 255), 2)
        
        # Draw assigned pieces
        for i, piece in enumerate(pieces):
            if piece is not None:
                # Convert square index to rank/file for warped board
                rank = 7 - (i // 8)
                file = i % 8
                
                # Calculate position on warped board
                x = file * square_size + square_size // 2
                y = rank * square_size + square_size // 2
                
                # Choose color based on piece color
                piece_color = (0, 255, 255) if piece.symbol().islower() else (255, 255, 0)  # Bright colors for visibility
                
                # Draw piece symbol with background for visibility
                symbol = piece.symbol()
                text_size = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                cv2.rectangle(viz, (x - text_size[0]//2 - 5, y - text_size[1] - 5), 
                            (x + text_size[0]//2 + 5, y + 5), (0, 0, 0), -1)
                cv2.putText(viz, symbol, (x - text_size[0]//2, y), cv2.FONT_HERSHEY_SIMPLEX, 2, piece_color, 3)
                
                # Draw confidence
                confidence = getattr(piece, '_confidence', 0.0)
                cv2.putText(viz, f"{confidence:.2f}", (x - 20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, piece_color, 1)
        
        return viz
    
    def _visualize_clean_final_board(self, pieces):
        """Create a clean visualization of the final board state."""
        # Create a chess board visualization
        canvas_size = 600
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        square_size = canvas_size // 8
        
        # Draw chessboard pattern
        for rank in range(8):
            for file in range(8):
                if (rank + file) % 2 == 1:  # Dark squares
                    x1 = file * square_size
                    y1 = rank * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (181, 136, 99), -1)
        
        # Draw coordinate labels
        for i in range(8):
            # File labels (a-h)
            file_label = chr(ord('a') + i)
            cv2.putText(canvas, file_label, (i * square_size + square_size//2 - 8, canvas_size - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Rank labels (1-8)
            rank_label = str(8 - i)
            cv2.putText(canvas, rank_label, (10, i * square_size + square_size//2 + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw pieces with nice Unicode symbols
        piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        for i, piece in enumerate(pieces):
            if piece is not None:
                rank = 7 - (i // 8)
                file = i % 8
                
                x = file * square_size + square_size // 2
                y = rank * square_size + square_size // 2
                
                symbol = piece.symbol()
                unicode_symbol = piece_symbols.get(symbol, symbol)
                
                # Use different colors for white/black pieces
                color = (50, 50, 50) if symbol.isupper() else (150, 50, 50)
                
                # Draw the piece with good visibility
                cv2.putText(canvas, unicode_symbol, (x - 25, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
        
        # Add title
        num_pieces = np.count_nonzero(pieces != None)
        title = f"Final Chess Board ({num_pieces} pieces)"
        cv2.putText(canvas, title, (canvas_size//2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return canvas
    
    def _classify_pieces_yolo_debug(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray,
                                   custom_conf_threshold=None, custom_iou_threshold=None, use_original_image=False):
        """Classify pieces using YOLO detection, returning both pieces and raw detections."""
        from chesscog.piece_classifier.yolo_utils import detect_pieces_with_yolo, detect_pieces_with_yolo_on_original
        
        # Get YOLO configuration parameters with custom overrides
        confidence_threshold = custom_conf_threshold or getattr(self._pieces_cfg.YOLO, 'CONFIDENCE_THRESHOLD', 0.5)
        iou_threshold = custom_iou_threshold or getattr(self._pieces_cfg.YOLO, 'IOU_THRESHOLD', 0.4)
        
        # Choose detection method
        if use_original_image:
            # Run YOLO on original image, then transform detections
            pieces_array, raw_detections = detect_pieces_with_yolo_on_original(
                self._pieces_model, img, corners, confidence_threshold, iou_threshold
            )
        else:
            # Run YOLO on warped image with improved mapping
            pieces_array, raw_detections = detect_pieces_with_yolo(
                self._pieces_model, img, corners, confidence_threshold, iou_threshold
            )
        
        # Add class names to raw detections for better display
        class_names = [
            "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
            "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
        ]
        for detection in raw_detections:
            class_id = detection['class']
            if class_id < len(class_names):
                detection['class_name'] = class_names[class_id]
        
        return pieces_array, raw_detections

    def _visualize_yolo_results(self, img, corners, pieces, raw_detections=None, approach_name="Unknown"):
        """Visualize YOLO detection results with bounding boxes."""
        viz = img.copy()
        
        # Draw detected corners first
        if corners is not None:
            corners_viz = self._visualize_corners(viz, corners)
            viz = corners_viz
            
        # Create visualization based on approach
        h, w = viz.shape[:2]
        
        if approach_name == "Original Image":
            # For original image approach, just show the original image with detections
            combined = viz.copy()
            detection_area_offset_x = 0
            detection_scale = 1.0
        else:
            # For warped image approach, show original + warped side by side
            from chesscog.piece_classifier.yolo_utils import warp_board_for_yolo
            warped_board = warp_board_for_yolo(img, corners)
            
            warped_h, warped_w = warped_board.shape[:2]
            
            # Scale warped to fit alongside original
            scale = min(h / warped_h, w / warped_w * 0.4)  # Make warped board smaller
            new_h = int(warped_h * scale)
            new_w = int(warped_w * scale)
            warped_scaled = cv2.resize(warped_board, (new_w, new_h))
            
            # Create side-by-side visualization
            combined = np.zeros((h, w + new_w + 20, 3), dtype=np.uint8)
            combined[:h, :w] = viz
            combined[:new_h, w+20:w+20+new_w] = warped_scaled
            
            detection_area_offset_x = w + 20
            detection_scale = scale
        
        # Draw raw YOLO detections on warped board section
        if raw_detections:
            class_names = [
                "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
                "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
            ]
            
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
            ]
            
            for detection in raw_detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_id = detection['class']
                
                if class_id < len(class_names):
                    # Scale bounding box based on approach
                    x1 = int(bbox[0] * detection_scale) + detection_area_offset_x
                    y1 = int(bbox[1] * detection_scale)
                    x2 = int(bbox[2] * detection_scale) + detection_area_offset_x
                    y2 = int(bbox[3] * detection_scale)
                    
                    # Draw bounding box
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    piece_name = class_names[class_id]
                    label = f"{piece_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(combined, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(combined, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
        # Add labels for the sections
        if approach_name == "Original Image":
            cv2.putText(combined, f"{approach_name} + YOLO Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(combined, "Original + Corners", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Warped + YOLO Detections", (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add final piece assignments at bottom
        y_offset = h - 150
        x_offset = 10
        cv2.putText(combined, "Final Pieces:", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30
        
        for i, piece in enumerate(pieces):
            if piece is not None:
                square_name = chess.square_name(i)
                text = f"{square_name}: {piece.symbol()}"
                cv2.putText(combined, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                
                if y_offset > h - 30:  # Wrap to next column
                    y_offset = h - 120
                    x_offset += 150
                    
        return combined

# Global recognizer instance
recognizer = None

def get_recognizer():
    """Get or create the debugging chess recognizer."""
    global recognizer
    if recognizer is None:
        try:
            recognizer = DebuggingChessRecognizer()
        except Exception as e:
            print(f"Failed to initialize recognizer: {e}")
            return None
    return recognizer

@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image through the chess recognition pipeline."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        img_data = file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get recognizer
        recog = get_recognizer()
        if recog is None:
            return jsonify({'error': 'Failed to initialize chess recognizer'}), 500
        
        # Get threshold parameters
        custom_conf = request.form.get('confidence_threshold', type=float)
        custom_iou = request.form.get('iou_threshold', type=float)
        use_original_image = request.form.get('use_original_image') == 'true'
        
        # Run debug analysis
        turn = chess.WHITE if request.form.get('turn', 'white') == 'white' else chess.BLACK
        results = recog.predict_with_debug(img, turn, custom_conf, custom_iou, use_original_image)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChessCog Debug Visualizer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-form {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        input[type="file"] {
            flex: 1;
            min-width: 200px;
        }
        button {
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
        }
        .turn-selector, .yolo-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .yolo-controls input[type="number"] {
            width: 70px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .results {
            display: none;
            margin-top: 20px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 5px solid #c62828;
        }
        .step {
            background: white;
            margin: 20px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .step-header {
            background: #2196F3;
            color: white;
            padding: 15px;
            font-weight: bold;
            font-size: 18px;
        }
        .step-content {
            padding: 20px;
        }
        .step-description {
            margin-bottom: 15px;
            color: #666;
        }
        .step-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metadata {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
        .final-result {
            background: #e8f5e8;
            border: 2px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .chess-board {
            font-family: monospace;
            font-size: 14px;
            line-height: 1.2;
            background: white;
            padding: 15px;
            border-radius: 5px;
            white-space: pre;
            overflow-x: auto;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ChessCog Pipeline Debug Visualizer</h1>
        <p>Upload a chess board image to see step-by-step analysis results</p>
    </div>
    
    <div class="upload-section">
        <form id="uploadForm" class="upload-form">
            <input type="file" id="imageFile" accept="image/*" required>
            <div class="turn-selector">
                <label>Turn:</label>
                <label><input type="radio" name="turn" value="white" checked> White</label>
                <label><input type="radio" name="turn" value="black"> Black</label>
            </div>
            <div class="yolo-controls">
                <label>YOLO Confidence: <input type="number" name="confidence_threshold" min="0.1" max="1.0" step="0.05" value="0.25" placeholder="0.25"></label>
                <label>YOLO IoU: <input type="number" name="iou_threshold" min="0.1" max="1.0" step="0.05" value="0.4" placeholder="0.4"></label>
                <label><input type="checkbox" name="use_original_image"> Use Original Image (vs Warped)</label>
            </div>
            <button type="submit" id="analyzeBtn">Analyze Image</button>
        </form>
    </div>
    
    <div id="results" class="results"></div>
    
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const results = document.getElementById('results');
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            results.style.display = 'block';
            results.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Running chess recognition pipeline...</p>
                </div>
            `;
            
            try {
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                formData.append('turn', document.querySelector('input[name="turn"]:checked').value);
                
                // Add YOLO threshold parameters
                const confThreshold = document.querySelector('input[name="confidence_threshold"]').value;
                const iouThreshold = document.querySelector('input[name="iou_threshold"]').value;
                const useOriginalImage = document.querySelector('input[name="use_original_image"]').checked;
                if (confThreshold) formData.append('confidence_threshold', confThreshold);
                if (iouThreshold) formData.append('iou_threshold', iouThreshold);
                formData.append('use_original_image', useOriginalImage);
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    results.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                } else {
                    displayResults(data);
                }
                
            } catch (error) {
                results.innerHTML = `<div class="error">Request failed: ${error.message}</div>`;
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Image';
            }
        });
        
        function displayResults(data) {
            const results = document.getElementById('results');
            let html = '';
            
            if (data.error) {
                html += `<div class="error">Pipeline Error: ${data.error}</div>`;
            }
            
            // Show each step
            data.steps.forEach((step, index) => {
                let metadataHtml = '';
                if (step.metadata) {
                    if (step.name.includes('Detection → Board Mapping') && step.metadata.assignment_details) {
                        // Special formatting for mapping step
                        const assignments = step.metadata.assignment_details;
                        const filteredMeta = {...step.metadata};
                        delete filteredMeta.assignment_details;
                        
                        metadataHtml = `
                            <div class="metadata">
                                ${JSON.stringify(filteredMeta, null, 2)}
                                <hr>
                                <h4>Piece Assignments (${assignments.length}):</h4>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 10px;">
                                    ${assignments.map(assignment => `
                                        <div style="background: #e8f5e8; padding: 8px; border-radius: 3px; border-left: 3px solid #4CAF50;">
                                            <strong>${assignment.square_name}:</strong> ${assignment.piece_symbol}<br>
                                            <small>${assignment.piece_name} (${assignment.confidence.toFixed(3)})</small>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    } else if (step.name.includes('YOLO') && step.metadata.raw_detections) {
                        // Special formatting for YOLO detections
                        const detections = step.metadata.raw_detections;
                        const filteredMeta = {...step.metadata};
                        delete filteredMeta.raw_detections;
                        
                        metadataHtml = `
                            <div class="metadata">
                                ${JSON.stringify(filteredMeta, null, 2)}
                                <hr>
                                <h4>Raw YOLO Detections (${detections.length}):</h4>
                                ${detections.slice(0, 10).map((det, i) => `
                                    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                        <strong>Detection ${i+1}:</strong> ${det.class_name || 'class_' + det.class}<br>
                                        <strong>Confidence:</strong> ${det.confidence.toFixed(3)}<br>
                                        <strong>BBox:</strong> [${det.bbox.map(b => b.toFixed(1)).join(', ')}]
                                    </div>
                                `).join('')}
                                ${detections.length > 10 ? `<p><em>... and ${detections.length - 10} more detections</em></p>` : ''}
                            </div>
                        `;
                    } else {
                        metadataHtml = `<div class="metadata">${JSON.stringify(step.metadata, null, 2)}</div>`;
                    }
                }
                
                html += `
                    <div class="step">
                        <div class="step-header">${step.name}</div>
                        <div class="step-content">
                            <div class="step-description">${step.description}</div>
                            <img src="${step.image}" alt="${step.name}" class="step-image">
                            ${metadataHtml}
                        </div>
                    </div>
                `;
            });
            
            // Show final result
            if (data.final_board && !data.error) {
                html += `
                    <div class="final-result">
                        <h3>Final Result</h3>
                        <p><strong>FEN:</strong> ${data.final_board}</p>
                        ${data.board_unicode ? `<div class="chess-board">${data.board_unicode}</div>` : ''}
                        ${data.final_corners ? `<div class="metadata">Corners: ${JSON.stringify(data.final_corners, null, 2)}</div>` : ''}
                    </div>
                `;
            }
            
            results.innerHTML = html;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting ChessCog Debug Visualizer...")
    print("Make sure you have all the required models in the models:// directory")
    app.run(debug=True, host='0.0.0.0', port=5000)
