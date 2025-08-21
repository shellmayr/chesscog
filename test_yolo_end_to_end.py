#!/usr/bin/env python3
"""
End-to-end test script for YOLO chess piece detection.

This script tests the complete pipeline:
1. YOLO model loading
2. Corner detection  
3. Piece detection using YOLO
4. Board reconstruction
5. FEN generation

Usage:
    python test_yolo_end_to_end.py [image_path]
"""

import sys
import cv2
import numpy as np
import chess
from pathlib import Path
import time
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, '.')

from chesscog.recognition.recognition import ChessRecognizer, TimedChessRecognizer


def test_yolo_integration():
    """Test YOLO model loading and integration."""
    print("🧪 Testing YOLO Integration...")
    print("-" * 40)
    
    try:
        recognizer = ChessRecognizer()
        
        print(f"✅ ChessRecognizer loaded")
        print(f"🤖 Using YOLO: {recognizer._use_yolo}")
        
        if not recognizer._use_yolo:
            print("❌ YOLO not enabled - check configuration")
            return False
            
        # Test model loading
        model_path = getattr(recognizer._pieces_cfg.TRAINING.MODEL, 'MODEL_PATH', 'Not set')
        print(f"📁 Model path: {model_path}")
        
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
            
        print(f"✅ Model file exists ({model_file.stat().st_size / (1024*1024):.1f} MB)")
        
        # Verify model is loaded
        if recognizer._pieces_model.model is None:
            print("❌ YOLO model not loaded")
            return False
            
        print(f"✅ YOLO model loaded: {type(recognizer._pieces_model.model).__name__}")
        print(f"🎯 Settings: conf={recognizer._pieces_cfg.YOLO.CONFIDENCE_THRESHOLD}, iou={recognizer._pieces_cfg.YOLO.IOU_THRESHOLD}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_recognition(image_path: str, perspective: chess.Color = chess.WHITE):
    """Test end-to-end chess recognition on an image."""
    print(f"\n🖼️  Testing Image Recognition: {image_path}")
    print("-" * 60)
    
    # Load image
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
        
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {img.shape}")
    
    try:
        # Use timed recognizer for detailed timing
        recognizer = TimedChessRecognizer()
        
        print("🔍 Performing recognition...")
        board, corners, times = recognizer.predict(img, perspective)
        
        # Display results
        print(f"✅ Recognition completed!")
        print(f"⏱️  Timing breakdown:")
        for stage, duration in times.items():
            print(f"   {stage}: {duration:.3f}s")
        print(f"   Total: {sum(times.values()):.3f}s")
            
        print(f"\n🏁 Results:")
        print(f"📋 Detected position:")
        print(board)
        
        print(f"\n🔤 FEN: {board.fen()}")
        print(f"🌐 View online: https://lichess.org/editor/{board.board_fen()}")
        
        # Check position validity
        if board.status() == chess.Status.VALID:
            print("✅ Position is legal")
        else:
            print("⚠️  Position may not be legal (this is normal for partial boards or detection errors)")
            
        # Count pieces
        piece_count = len([p for p in board.piece_map().values()])
        print(f"🎯 Detected {piece_count} pieces")
        
        return True
        
    except Exception as e:
        print(f"❌ Recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end YOLO chess recognition test")
    parser.add_argument("image", nargs="?", help="Path to chess board image", 
                       default="images/opening.png")
    parser.add_argument("--white", action="store_true", dest="perspective", 
                       help="Image from white's perspective (default)")
    parser.add_argument("--black", action="store_false", dest="perspective",
                       help="Image from black's perspective")
    parser.set_defaults(perspective=True)
    
    args = parser.parse_args()
    
    print("🚀 YOLO Chess Recognition - End-to-End Test")
    print("=" * 60)
    
    # Test integration first
    if not test_yolo_integration():
        print("\n❌ Integration test failed - cannot proceed")
        return 1
    
    # Test image recognition
    perspective = chess.WHITE if args.perspective else chess.BLACK
    if not test_image_recognition(args.image, perspective):
        print("\n❌ Image recognition test failed")
        return 1
    
    print("\n🎉 All tests completed successfully!")
    print("Your YOLO chess recognition system is ready to use!")
    
    return 0


if __name__ == "__main__":
    exit(main())
