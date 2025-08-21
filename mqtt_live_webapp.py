#!/usr/bin/env python3
"""
Live MQTT Message Viewer Web App
Subscribes to MQTT topics and displays messages in real-time using Server-Sent Events.
"""

import os
import ssl
import json
import base64
import binascii
import threading
import queue
import sys
import time
import atexit
from datetime import datetime
from collections import deque
import numpy as np
import cv2
import chess
import torch
from flask import Flask, render_template_string, Response, jsonify
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# ChessCog imports
try:
    from chesscog.recognition.recognition import ChessRecognizer
    from chesscog.corner_detection import find_corners, resize_image
    from chesscog.core.exceptions import ChessboardNotLocatedException
    CHESS_RECOGNITION_AVAILABLE = True
    print("✅ Chess recognition modules loaded successfully")
except ImportError as e:
    CHESS_RECOGNITION_AVAILABLE = False
    print(f"⚠️ Chess recognition not available: {e}")
    print("Images will be displayed without recognition analysis")

load_dotenv()

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

class MQTTChessRecognizer:
    """Chess recognizer for processing MQTT images."""
    
    def __init__(self):
        self.recognizer = None
        self.available = False
        
        if CHESS_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = ChessRecognizer()
                self.available = True
                print("✅ Chess recognizer initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize chess recognizer: {e}")
                self.available = False
    
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
            cv2.circle(viz, tuple(corner), 8, (255, 0, 0), 2)
            cv2.putText(viz, str(i), tuple(corner + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw bounding rectangle
        cv2.polylines(viz, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        
        return viz
    
    def _visualize_final_board(self, pieces):
        """Create a visualization of the final board state."""
        # Create a chess board visualization
        canvas_size = 400
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
            cv2.putText(canvas, file_label, (i * square_size + square_size//2 - 5, canvas_size - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Rank labels (1-8)
            rank_label = str(8 - i)
            cv2.putText(canvas, rank_label, (8, i * square_size + square_size//2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw pieces with Unicode symbols
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
                cv2.putText(canvas, unicode_symbol, (x - 12, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        return canvas
    
    def process_image(self, image_data, turn=chess.WHITE):
        """Process image through chess recognition pipeline."""
        if not self.available:
            return {
                'success': False,
                'error': 'Chess recognition not available',
                'original_image': f"data:image/jpeg;base64,{image_data}"
            }
        
        try:
            # Decode base64 image
            print("🔄 Decoding base64 image...")
            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    'success': False,
                    'error': 'Failed to decode image data',
                    'original_image': f"data:image/jpeg;base64,{image_data}"
                }
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"📐 Image shape: {img.shape}")
            
            # Run recognition using the same approach as debug_webapp
            print("♟️ Running full chess recognition...")
            with torch.no_grad():
                try:
                    # Use the same approach as debug_webapp - just call predict() directly
                    board, final_corners = self.recognizer.predict(img, turn)
                    print(f"✅ Recognition successful, board: {board.fen()}")
                    
                    # Create visualizations for the successful recognition
                    # Get the resized image and corners for visualization
                    img_resized, img_scale = resize_image(self.recognizer._corner_detection_cfg, img)
                    corners = find_corners(self.recognizer._corner_detection_cfg, img_resized)
                    corners_viz = self._visualize_corners(img_resized, corners)
                    
                    # Create board visualization
                    pieces = []
                    for square in self.recognizer._squares:
                        piece = board.piece_at(square)
                        pieces.append(piece)
                    
                    board_viz = self._visualize_final_board(pieces)
                    
                    return {
                        'success': True,
                        'board_fen': board.fen(),
                        'board_unicode': str(board),
                        'piece_count': len([p for p in pieces if p is not None]),
                        'corners': make_json_serializable(final_corners),
                        'images': {
                            'original': f"data:image/jpeg;base64,{image_data}",
                            'corners_detected': self._encode_image(corners_viz),
                            'board_state': self._encode_image(board_viz)
                        }
                    }
                    
                except ChessboardNotLocatedException as e:
                    print(f"📍 No chessboard found in image: {str(e)}")
                    return {
                        'success': False,
                        'error': 'No chess board detected in image',
                        'error_type': 'no_board',
                        'original_image': f"data:image/jpeg;base64,{image_data}"
                    }
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Recognition error: {error_msg}")
                    
                    # Check for common "no chess board" error patterns
                    no_board_patterns = [
                        'infinity', 'divide by zero', 'float32', 
                        'NoneType', 'squeeze', 'attribute', 
                        'has no attribute', 'cannot squeeze'
                    ]
                    
                    if any(pattern in error_msg for pattern in no_board_patterns):
                        print(f"📍 Treating as no chess board detected")
                        return {
                            'success': False,
                            'error': 'No chess board detected in image',
                            'error_type': 'no_board',
                            'original_image': f"data:image/jpeg;base64,{image_data}"
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Chess recognition failed: {str(e)}',
                            'error_type': 'processing_error',
                            'original_image': f"data:image/jpeg;base64,{image_data}"
                        }
                    
        except Exception as e:
            print(f"❌ Recognition error: {str(e)}")
            return {
                'success': False,
                'error': f"Recognition failed: {str(e)}",
                'original_image': f"data:image/jpeg;base64,{image_data}"
            }

# Global chess recognizer
chess_recognizer = MQTTChessRecognizer()

# MQTT Configuration
MQTT_URL = os.getenv('MQTT_URL')
MQTT_HOST = os.getenv('MQTT_HOST')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_INSECURE = os.getenv('MQTT_INSECURE', 'false').lower() == 'true'
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/hypervision/forka/device/10:51:DB:85:4B:B0/feed')
MQTT_CLIENT_ID = os.getenv('MQTT_CLIENT_ID', 'chess_mqtt_webapp_client')

# Message storage for live updates
MAX_MESSAGES = 100
message_queue = deque(maxlen=MAX_MESSAGES)
clients = set()  # Set to track connected SSE clients

class MQTTHandler:
    def __init__(self):
        self.client = None
        self.connected = False
        self.reconnect_count = 0
        self.max_reconnect_attempts = 5
        self.connection_lock = threading.Lock()
        self.should_connect = True
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            broker_info = MQTT_URL if MQTT_URL else f"{MQTT_HOST}:{MQTT_PORT}"
            print(f"✅ Connected to MQTT broker: {broker_info}")
            client.subscribe(MQTT_TOPIC)
            print(f"📡 Subscribed to topic: {MQTT_TOPIC}")
            self.connected = True
            self.reconnect_count = 0  # Reset reconnect counter on successful connection
            
            # Add connection status message
            status_message = {
                'timestamp': datetime.now().isoformat(),
                'topic': 'system',
                'type': 'status',
                'message': 'Connected to MQTT broker',
                'raw_message': 'System ready'
            }
            self.add_message(status_message)
        else:
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")
            self.connected = False
            
            # Add error status message
            error_message = {
                'timestamp': datetime.now().isoformat(),
                'topic': 'system',
                'type': 'error',
                'message': 'Connection failed',
                'raw_message': f'Return code: {rc}'
            }
            self.add_message(error_message)

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        message = msg.payload.decode('utf-8')
        timestamp = datetime.now()
        
        print(f"Received message on topic '{topic}': {message}")
        
        # Create message object
        msg_obj = {
            'timestamp': timestamp.isoformat(),
            'topic': topic,
            'raw_message': message,
            'message': message,
            'type': 'message'
        }
        
        try:
            # Try to parse as JSON
            data = json.loads(message)
            msg_obj['parsed_json'] = data
            msg_obj['type'] = 'json'
            
            # Check if this is an image message
            if data.get("type") == "image" and "image" in data:
                msg_obj['type'] = 'image'
                msg_obj['image_data'] = data["image"]
                
                # Save image to file (like original mqtt_subscriber.py)
                try:
                    image_data = base64.b64decode(data["image"])
                    os.makedirs("images", exist_ok=True)
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"images/image_{timestamp_str}.jpg"
                    
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    
                    msg_obj['saved_filename'] = filename
                    print(f"Saved image as: {filename}")
                    
                    # Process through chess recognition pipeline
                    print("🔍 Processing image through chess recognition...")
                    recognition_results = chess_recognizer.process_image(data["image"])
                    msg_obj['recognition_results'] = recognition_results
                    
                    if recognition_results['success']:
                        print(f"✅ Chess recognition successful - detected {recognition_results['piece_count']} pieces")
                        print(f"📋 Board FEN: {recognition_results['board_fen']}")
                    else:
                        print(f"❌ Chess recognition failed: {recognition_results.get('error', 'Unknown error')}")
                    
                except (binascii.Error, Exception) as e:
                    print(f"Error processing image: {e}")
                    msg_obj['error'] = f"Error processing image: {e}"
                    
        except json.JSONDecodeError:
            # Not a JSON message, keep as plain text
            pass
        except Exception as e:
            msg_obj['error'] = f"Error processing message: {e}"
            print(f"Unexpected error handling message: {e}")
        
        # Add message to queue and notify clients
        self.add_message(msg_obj)

    def on_disconnect(self, client, userdata, rc):
        print(f"🔌 Disconnected from MQTT broker (rc: {rc})")
        self.connected = False
        
        # Add disconnection status message
        disconnect_reason = "Connection closed" if rc == 0 else "Connection lost"
        status_message = {
            'timestamp': datetime.now().isoformat(),
            'topic': 'system',
            'type': 'status' if rc == 0 else 'error',
            'message': disconnect_reason,
            'raw_message': f'Return code: {rc}' if rc != 0 else ''
        }
        self.add_message(status_message)

    def add_message(self, msg_obj):
        """Add message to queue and notify all connected SSE clients."""
        message_queue.append(msg_obj)
        
        # Notify all connected clients
        sse_data = f"data: {json.dumps(msg_obj)}\n\n"
        disconnected_clients = set()
        
        for client_queue in clients:
            try:
                client_queue.put(sse_data)
            except Exception:
                disconnected_clients.add(client_queue)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            clients.discard(client)

    def connect(self):
        """Initialize and connect MQTT client."""
        with self.connection_lock:
            if self.client is not None and self.connected:
                print("🔄 MQTT client already connected, skipping...")
                return True
            
            # Clean up existing client if any
            if self.client is not None:
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                except Exception:
                    pass
            
            # Determine broker connection details (from original mqtt_subscriber.py)
            if MQTT_URL:
                if MQTT_URL.startswith('mqtts://'):
                    broker_host = MQTT_URL.replace('mqtts://', '').split(':')[0]
                    broker_port = int(MQTT_URL.split(':')[-1]) if ':' in MQTT_URL.replace('mqtts://', '') else 8883
                    use_ssl = True
                elif MQTT_URL.startswith('mqtt://'):
                    broker_host = MQTT_URL.replace('mqtt://', '').split(':')[0]
                    broker_port = int(MQTT_URL.split(':')[-1]) if ':' in MQTT_URL.replace('mqtt://', '') else 1883
                    use_ssl = False
                else:
                    broker_host = MQTT_URL.split(':')[0]
                    broker_port = int(MQTT_URL.split(':')[1]) if ':' in MQTT_URL else MQTT_PORT
                    use_ssl = not MQTT_INSECURE
            else:
                broker_host = MQTT_HOST
                broker_port = MQTT_PORT
                use_ssl = not MQTT_INSECURE

            if not all([broker_host, MQTT_USERNAME, MQTT_PASSWORD]):
                error_msg = "❌ Error: Missing required MQTT credentials in .env file"
                print(error_msg)
                return False

            # Create new client with clean session
            self.client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
            
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            if use_ssl:
                if MQTT_INSECURE:
                    # Disable SSL certificate verification for self-signed certificates
                    self.client.tls_set(cert_reqs=ssl.CERT_NONE)
                    self.client.tls_insecure_set(True)
                else:
                    self.client.tls_set()
            
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect

            try:
                connection_type = "secure" if use_ssl else "insecure"
                print(f"🔄 Connecting to MQTT broker at {broker_host}:{broker_port} ({connection_type})...")
                
                # Disable automatic reconnect to prevent connection loops
                self.client.reconnect_delay_set(min_delay=5, max_delay=300)
                
                # Connect with more conservative keepalive
                self.client.connect(broker_host, broker_port, keepalive=60)
                self.client.loop_start()
                return True
            except Exception as e:
                print(f"❌ Failed to connect to MQTT broker: {e}")
                return False
    
    def disconnect(self):
        """Cleanly disconnect MQTT client."""
        with self.connection_lock:
            self.should_connect = False
            if self.client is not None:
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                    print("🔌 MQTT client disconnected cleanly")
                except Exception as e:
                    print(f"⚠️ Error during MQTT disconnect: {e}")
                finally:
                    self.client = None
                    self.connected = False

# Global MQTT handler
mqtt_handler = MQTTHandler()

def start_mqtt_client():
    """Start MQTT client in background thread."""
    try:
        success = mqtt_handler.connect()
        if not success:
            print("❌ Failed to start MQTT client")
    except Exception as e:
        print(f"❌ Exception while starting MQTT client: {e}")

def cleanup_mqtt():
    """Cleanup MQTT connection on shutdown."""
    print("🧹 Cleaning up MQTT connection...")
    mqtt_handler.disconnect()

# Register cleanup function
atexit.register(cleanup_mqtt)

# Server-Sent Events endpoint
@app.route('/events')
def events():
    """Server-Sent Events endpoint for real-time updates."""
    
    def event_stream():
        client_queue = queue.Queue()
        clients.add(client_queue)
        
        try:
            # Send existing messages to new client
            for msg in message_queue:
                yield f"data: {json.dumps(msg)}\n\n"
            
            # Stream new messages
            while True:
                try:
                    data = client_queue.get(timeout=30)  # 30 second timeout for keepalive
                    yield data
                except queue.Empty:
                    yield "data: {\"type\": \"keepalive\"}\n\n"  # Keepalive ping
                    
        except GeneratorExit:
            clients.discard(client_queue)
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/status')
def status():
    """Get current MQTT connection status."""
    return jsonify({
        'connected': mqtt_handler.connected,
        'topic': MQTT_TOPIC,
        'message_count': len(message_queue),
        'active_clients': len(clients)
    })

@app.route('/messages')
def get_messages():
    """Get recent messages as JSON."""
    return jsonify(list(message_queue))

@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live MQTT Message Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #fafafa;
            height: 100vh;
            overflow: hidden;
            color: #0a0a0a;
        }
        .header {
            background: white;
            padding: 16px 24px;
            border-bottom: 1px solid #e4e4e7;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            height: 64px;
            box-sizing: border-box;
        }
        .header h1 {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            color: #09090b;
            letter-spacing: -0.025em;
        }
        .header p {
            margin: 4px 0 0 0;
            font-size: 11px;
            color: #71717a;
            font-weight: 400;
        }
        .status-bar {
            background: white;
            padding: 12px 24px;
            border-bottom: 1px solid #e4e4e7;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            position: fixed;
            top: 64px;
            left: 0;
            right: 0;
            z-index: 999;
            height: 48px;
            box-sizing: border-box;
            font-size: 12px;
        }
        .main-container {
            display: flex;
            height: 100vh;
            padding-top: 112px;
            box-sizing: border-box;
        }
        .feed-panel {
            width: 33.333%;
            background: white;
            border-right: 1px solid #e4e4e7;
            overflow-y: auto;
            height: calc(100vh - 112px);
        }
        .image-panel {
            width: 66.666%;
            background: #f9fafb;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 112px);
            overflow: hidden;
            position: relative;
        }
        .image-header {
            background: white;
            padding: 16px 24px;
            border-bottom: 1px solid #e4e4e7;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .image-title {
            font-weight: 600;
            font-size: 14px;
            color: #09090b;
        }
        .image-controls {
            display: flex;
            gap: 8px;
        }
        .image-tab {
            padding: 6px 12px;
            background: #f1f5f9;
            color: #64748b;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.15s ease;
        }
        .image-tab.active {
            background: #09090b;
            color: white;
        }
        .image-tab:hover:not(.active) {
            background: #e2e8f0;
        }
        .image-content {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 24px;
        }
        .status-indicator {
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        .status-connected {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .status-disconnected {
            background: #fef2f2;
            color: #dc2626;
            border: 1px solid #fecaca;
        }
        .status-info {
            color: #71717a;
            font-size: 11px;
            font-weight: 400;
        }
        .feed-header {
            background: #09090b;
            color: white;
            padding: 16px 20px;
            font-weight: 600;
            font-size: 13px;
            position: sticky;
            top: 0;
            z-index: 10;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #27272a;
        }
        .message {
            padding: 12px 20px;
            border-bottom: 1px solid #f4f4f5;
            transition: all 0.15s ease;
            font-size: 12px;
        }
        .message:hover {
            background-color: #f9fafb;
        }
        .latest-image {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e4e4e7;
        }
        .no-image {
            text-align: center;
            color: #71717a;
            font-size: 16px;
            font-weight: 400;
        }
        .image-info {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(9, 9, 11, 0.8);
            backdrop-filter: blur(8px);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 400;
            line-height: 1.4;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 280px;
            z-index: 10;
        }
        .board-info {
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: white;
            border: 1px solid #e4e4e7;
            border-radius: 8px;
            padding: 16px;
            font-size: 11px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            max-width: 300px;
            z-index: 10;
        }
        .board-info h4 {
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: 600;
            color: #09090b;
        }
        .board-info .fen {
            font-family: ui-monospace, monospace;
            background: #f9fafb;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #f1f5f9;
            margin: 8px 0;
            font-size: 10px;
            word-break: break-all;
            color: #374151;
        }
        .board-info .stats {
            display: flex;
            gap: 12px;
            margin-top: 8px;
            font-size: 10px;
            color: #71717a;
        }
        .message.new {
            background-color: #f0f9ff;
            animation: fadeIn 0.4s ease-out;
        }
        @keyframes fadeIn {
            from { background-color: #dbeafe; }
            to { background-color: #f0f9ff; }
        }
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }
        .message-title {
            font-weight: 500;
            color: #09090b;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .message-icon {
            width: 14px;
            height: 14px;
            flex-shrink: 0;
        }
        .message-timestamp {
            color: #71717a;
            font-size: 10px;
            font-weight: 400;
        }
        .message-description {
            color: #71717a;
            font-size: 11px;
            font-weight: 400;
            line-height: 1.4;
            margin-top: 4px;
        }
        .message-content {
            margin-top: 8px;
        }
        .message-text {
            background: #f9fafb;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 80px;
            overflow-y: auto;
            border: 1px solid #f1f5f9;
            color: #374151;
            line-height: 1.4;
        }
        .message-json {
            background: #f9fafb;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", monospace;
            font-size: 9px;
            overflow: hidden;
            max-height: 80px;
            position: relative;
            border: 1px solid #f1f5f9;
            color: #374151;
            line-height: 1.3;
        }
        .message-json::after {
            content: '';
            position: absolute;
            bottom: 0;
            right: 0;
            left: 0;
            height: 20px;
            background: linear-gradient(transparent, #f9fafb);
        }
        .message-image-preview {
            display: none;
        }
        .no-messages {
            padding: 48px 24px;
            text-align: center;
            color: #71717a;
            font-weight: 400;
        }
        .auto-scroll {
            background: #09090b;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
            font-size: 10px !important;
            padding: 6px 10px !important;
            font-weight: 500;
        }
        .auto-scroll:hover {
            background: #27272a;
        }
        .auto-scroll.disabled {
            background: #71717a;
            cursor: not-allowed;
        }
        .auto-scroll.disabled:hover {
            background: #71717a;
        }
        .loading {
            text-align: center;
            padding: 32px 24px;
            color: #71717a;
            font-weight: 400;
            font-size: 13px;
        }

        /* SVG Icons */
        .icon-camera {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2316a34a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z'/%3E%3Ccircle cx='12' cy='13' r='3'/%3E%3C/svg%3E");
        }
        .icon-play {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2306b6d4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='5,3 19,12 5,21'/%3E%3C/svg%3E");
        }
        .icon-stop {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='6' y='6' width='12' height='12' rx='2'/%3E%3C/svg%3E");
        }
        .icon-message {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/%3E%3C/svg%3E");
        }
        .icon-info {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23d97706' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='12' y1='16' x2='12' y2='12'/%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'/%3E%3C/svg%3E");
        }
        .icon-alert {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23dc2626' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/%3E%3Cline x1='12' y1='9' x2='12' y2='13'/%3E%3Cline x1='12' y1='17' x2='12.01' y2='17'/%3E%3C/svg%3E");
        }
        .icon-chess {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%237c3aed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M18 3H6l3 7 1.5-1.5L12 8l1.5.5L15 10l3-7z'/%3E%3Cpath d='M6 11h12v10H6z'/%3E%3Cpath d='M8 21h8'/%3E%3C/svg%3E");
        }

        .message-icon {
            width: 14px;
            height: 14px;
            flex-shrink: 0;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>MQTT Live Viewer</h1>
        <p>Real-time message feed with live image display</p>
    </div>
    
    <div class="status-bar">
        <div>
            <span id="status-indicator" class="status-indicator status-disconnected">Disconnected</span>
            <span id="topic-info" class="status-info">Topic: Loading...</span>
        </div>
        <div class="status-info">
            <span id="message-count">Messages: 0</span> | 
            <span id="client-count">Clients: 0</span>
        </div>
    </div>
    
    <div class="main-container">
        <div class="feed-panel">
            <div class="feed-header">
                Live Feed <span id="live-indicator" style="color: #22c55e;">●</span>
                <button id="auto-scroll" class="auto-scroll" onclick="toggleAutoScroll()">📍 ON</button>
            </div>
            <div id="messages">
                <div class="loading">Connecting to live feed...</div>
            </div>
        </div>
        
        <div class="image-panel">
            <div class="image-header">
                <div class="image-title">Chess Board Analysis</div>
                <div class="image-controls">
                    <button class="image-tab" onclick="showImageTab('original')">Original</button>
                    <button class="image-tab" onclick="showImageTab('corners')">Corners</button>
                    <button class="image-tab active" onclick="showImageTab('board')">Board</button>
                </div>
            </div>
            <div class="image-content">
                <div id="latest-image-container">
                    <div class="no-image">
                        <div style="font-size: 48px; margin-bottom: 16px;">♟️</div>
                        <div style="font-weight: 600; color: #09090b; margin-bottom: 8px;">Chess Board Analysis</div>
                        <div>Waiting for images to analyze...</div>
                    </div>
                </div>
                <div id="image-info" class="image-info" style="display: none;"></div>
                <div id="board-info" class="board-info" style="display: none;"></div>
            </div>
        </div>
    </div>
    
    <script>
        let autoScrollEnabled = true;
        let messageCount = 0;
        let latestImageData = null;
        let currentImageTab = 'board';  // Default to board view
        let recognitionResults = null;
        
        // Update status
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusEl = document.getElementById('status-indicator');
                    const topicEl = document.getElementById('topic-info');
                    const messageCountEl = document.getElementById('message-count');
                    const clientCountEl = document.getElementById('client-count');
                    
                    if (data.connected) {
                        statusEl.textContent = 'Connected';
                        statusEl.className = 'status-indicator status-connected';
                    } else {
                        statusEl.textContent = 'Disconnected';
                        statusEl.className = 'status-indicator status-disconnected';
                    }
                    
                    topicEl.textContent = `Topic: ${data.topic}`;
                    messageCountEl.textContent = `Messages: ${data.message_count}`;
                    clientCountEl.textContent = `Clients: ${data.active_clients}`;
                })
                .catch(err => {
                    console.error('Error updating status:', err);
                });
        }
        
        // Start Server-Sent Events
        function startEventStream() {
            const eventSource = new EventSource('/events');
            const messagesContainer = document.getElementById('messages');
            const liveIndicator = document.getElementById('live-indicator');
            
            eventSource.onopen = function() {
                console.log('Connected to event stream');
                messagesContainer.innerHTML = '<div class="no-messages">Waiting for activity...</div>';
                liveIndicator.style.color = '#22c55e';
            };
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'keepalive') {
                    return; // Ignore keepalive messages
                }
                
                // Remove "no messages" placeholder
                if (messageCount === 0) {
                    messagesContainer.innerHTML = '';
                }
                
                addMessage(data);
                messageCount++;
                updateStatus(); // Update status after new message
            };
            
            eventSource.onerror = function(event) {
                console.error('Event stream error:', event);
                liveIndicator.style.color = '#f44336';
                
                // Try to reconnect after 5 seconds
                setTimeout(() => {
                    if (eventSource.readyState === EventSource.CLOSED) {
                        location.reload(); // Reload page to restart connection
                    }
                }, 5000);
            };
        }
        
        function getMessageInfo(data) {
            // Parse message content and return user-friendly info
            let title = 'Message';
            let iconClass = 'icon-message';
            let description = '';
            
            if (data.type === 'image') {
                const filename = data.saved_filename ? data.saved_filename.split('/').pop() : 'image';
                
                if (data.recognition_results) {
                    if (data.recognition_results.success) {
                        title = 'Chess Board Analyzed';
                        iconClass = 'icon-camera';
                        description = `${filename} • ${data.recognition_results.piece_count} pieces detected`;
                    } else if (data.recognition_results.error_type === 'no_board') {
                        title = 'Image Received';
                        iconClass = 'icon-info';
                        description = `${filename} • No chess board detected`;
                    } else {
                        title = 'Image Received';
                        iconClass = 'icon-camera';
                        description = `${filename} • Analysis failed`;
                    }
                } else {
                    title = 'Image Received';
                    iconClass = 'icon-camera';
                    description = filename;
                }
            } else if (data.parsed_json) {
                const json = data.parsed_json;
                
                if (json.type === 'game_start') {
                    title = 'Game Started';
                    iconClass = 'icon-play';
                    description = 'New chess game initiated';
                } else if (json.type === 'game_end') {
                    title = 'Game Ended';
                    iconClass = 'icon-stop';
                    const player = json.player !== undefined ? `Player ${json.player}` : '';
                    description = player ? `Winner: ${player}` : 'Game completed';
                } else if (json.type === 'move' || json.move) {
                    title = 'Move Played';
                    iconClass = 'icon-chess';
                    description = `${json.move || 'Chess move'}`;
                } else if (json.type === 'status') {
                    title = 'Status Update';
                    iconClass = 'icon-info';
                    description = json.message || json.status || 'System status';
                } else {
                    title = 'Data Received';
                    iconClass = 'icon-message';
                    description = Object.keys(json).join(', ');
                }
            } else if (data.type === 'status') {
                title = 'Status';
                iconClass = 'icon-info';
                description = data.message;
            } else if (data.type === 'error') {
                title = 'Error';
                iconClass = 'icon-alert';
                description = data.message;
            } else {
                title = 'Message';
                iconClass = 'icon-message';
                description = data.message;
            }
            
            return { title, iconClass, description };
        }

        function addMessage(data) {
            const messagesContainer = document.getElementById('messages');
            const messageEl = document.createElement('div');
            messageEl.className = 'message new';
            
            const timestamp = new Date(data.timestamp).toLocaleString();
            const messageInfo = getMessageInfo(data);
            
            if (data.type === 'image') {
                // Update latest image in the right panel
                updateLatestImage(data);
            }
            
            // Create clean message display
            messageEl.innerHTML = `
                <div class="message-header">
                    <div class="message-title">
                        <div class="message-icon ${messageInfo.iconClass}"></div>
                        <span>${messageInfo.title}</span>
                    </div>
                    <span class="message-timestamp">${timestamp}</span>
                </div>
                ${messageInfo.description ? `<div class="message-description">${messageInfo.description}</div>` : ''}
            `;
            
            messagesContainer.appendChild(messageEl);
            
            // Auto-scroll if enabled - scroll the feed panel
            if (autoScrollEnabled) {
                const feedPanel = document.querySelector('.feed-panel');
                feedPanel.scrollTop = feedPanel.scrollHeight;
            }
            
            // Remove 'new' class after animation
            setTimeout(() => {
                messageEl.classList.remove('new');
            }, 1000);
        }
        
        function showImageTab(tab) {
            currentImageTab = tab;
            
            // Update tab buttons
            document.querySelectorAll('.image-tab').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[onclick="showImageTab('${tab}')"]`).classList.add('active');
            
            // Update image display
            if (latestImageData && recognitionResults) {
                updateImageDisplay();
            }
        }
        
        function updateImageDisplay() {
            const imageContainer = document.getElementById('latest-image-container');
            
            if (!latestImageData) return;
            
            let imageSource = '';
            let showBoardInfo = false;
            let showNoBoard = false;
            
            if (recognitionResults && recognitionResults.success) {
                switch (currentImageTab) {
                    case 'original':
                        imageSource = recognitionResults.images.original;
                        break;
                    case 'corners':
                        imageSource = recognitionResults.images.corners_detected;
                        break;
                    case 'board':
                        imageSource = recognitionResults.images.board_state;
                        showBoardInfo = true;
                        break;
                    default:
                        imageSource = recognitionResults.images.original;
                }
            } else if (recognitionResults && recognitionResults.error_type === 'no_board') {
                // Show original image with "no board detected" overlay
                imageSource = `data:image/jpeg;base64,${latestImageData.image_data}`;
                showNoBoard = true;
                
                // For board tab, show a special "no board" message
                if (currentImageTab === 'board') {
                    imageContainer.innerHTML = `
                        <div class="no-image">
                            <div style="text-align: center; color: #71717a; font-size: 16px; margin-bottom: 16px;">
                                <div style="font-size: 48px; margin-bottom: 12px;">🔍</div>
                                <div style="font-weight: 600; color: #09090b; margin-bottom: 8px;">No Chess Board Detected</div>
                                <div>The image doesn't contain a recognizable chess board</div>
                            </div>
                            <img src="${imageSource}" alt="Original Image" style="max-width: 300px; max-height: 300px; object-fit: contain; border-radius: 8px; opacity: 0.7;" />
                        </div>
                    `;
                    updateBoardInfo(false);
                    return;
                } else if (currentImageTab === 'corners') {
                    imageContainer.innerHTML = `
                        <div style="position: relative; display: inline-block;">
                            <img src="${imageSource}" alt="Original Image" class="latest-image" style="opacity: 0.7;" />
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(239, 68, 68, 0.9); color: white; padding: 16px 24px; border-radius: 8px; text-align: center; backdrop-filter: blur(4px);">
                                <div style="font-size: 20px; margin-bottom: 4px;">⚠️</div>
                                <div style="font-weight: 600; margin-bottom: 4px;">No Corners Detected</div>
                                <div style="font-size: 12px;">Chess board not found</div>
                            </div>
                        </div>
                    `;
                    updateBoardInfo(false);
                    return;
                }
            } else {
                // Fallback to original image for other errors
                imageSource = `data:image/jpeg;base64,${latestImageData.image_data}`;
            }
            
            let overlayHtml = '';
            if (showNoBoard && currentImageTab === 'original') {
                overlayHtml = `
                    <div style="position: absolute; top: 16px; left: 16px; background: rgba(239, 68, 68, 0.9); color: white; padding: 8px 12px; border-radius: 6px; font-size: 11px; backdrop-filter: blur(4px);">
                        ⚠️ No chess board detected
                    </div>
                `;
            }
            
            imageContainer.innerHTML = `
                <div style="position: relative; display: inline-block;">
                    <img src="${imageSource}" alt="Chess Board Analysis" class="latest-image" />
                    ${overlayHtml}
                </div>
            `;
            
            // Update board info
            updateBoardInfo(showBoardInfo);
        }
        
        function updateBoardInfo(show) {
            const boardInfo = document.getElementById('board-info');
            
            if (show && recognitionResults && recognitionResults.success) {
                boardInfo.innerHTML = `
                    <h4>Analysis Results</h4>
                    <div class="stats">
                        <span><strong>Pieces:</strong> ${recognitionResults.piece_count}</span>
                        <span><strong>Status:</strong> Valid Board</span>
                    </div>
                    <div class="fen">
                        <strong>FEN:</strong><br>
                        ${recognitionResults.board_fen}
                    </div>
                `;
                boardInfo.style.display = 'block';
            } else {
                boardInfo.style.display = 'none';
            }
        }
        
        function updateLatestImage(data) {
            const imageInfo = document.getElementById('image-info');
            
            if (data.image_data) {
                latestImageData = data;
                recognitionResults = data.recognition_results || null;
                
                // Update image display based on current tab
                updateImageDisplay();
                
                // Update image info overlay
                const timestamp = new Date(data.timestamp).toLocaleString();
                let infoContent = `<strong>Latest Image</strong><br>${timestamp}`;
                
                if (data.saved_filename) {
                    const filename = data.saved_filename.split('/').pop();
                    infoContent += `<br>File: ${filename}`;
                }
                
                if (recognitionResults) {
                    if (recognitionResults.success) {
                        infoContent += `<br><span style="color: #22c55e;">✓ Chess board detected</span>`;
                        infoContent += `<br><span style="color: #22c55e;">${recognitionResults.piece_count} pieces found</span>`;
                    } else if (recognitionResults.error_type === 'no_board') {
                        infoContent += `<br><span style="color: #f59e0b;">ℹ No chess board in image</span>`;
                    } else {
                        infoContent += `<br><span style="color: #ef4444;">✗ ${recognitionResults.error}</span>`;
                    }
                }
                
                imageInfo.innerHTML = infoContent;
                imageInfo.style.display = 'block';
            }
        }
        
        function toggleAutoScroll() {
            autoScrollEnabled = !autoScrollEnabled;
            const button = document.getElementById('auto-scroll');
            
            if (autoScrollEnabled) {
                button.textContent = '📍 ON';
                button.classList.remove('disabled');
            } else {
                button.textContent = '📍 OFF';
                button.classList.add('disabled');
            }
        }
        
        // Initialize
        updateStatus();
        startEventStream();
        
        // Update status every 10 seconds
        setInterval(updateStatus, 10000);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    
    # Check for production mode
    production_mode = '--production' in sys.argv or os.getenv('FLASK_ENV') == 'production'
    
    print("🚀 Starting Live MQTT Message Viewer...")
    print(f"📡 MQTT Topic: {MQTT_TOPIC}")
    print(f"🔧 Mode: {'Production' if production_mode else 'Development'}")
    print("⚙️ Make sure you have MQTT credentials configured in your .env file")
    
    # Start MQTT client in background thread
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    # Give MQTT client time to connect
    time.sleep(2)
    
    try:
        if production_mode:
            # Production mode: no debug, no reloader
            print("🌐 Starting in production mode on port 5001")
            app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
        else:
            # Development mode: debug enabled but no reloader to prevent MQTT reconnections
            print("🔧 Starting in development mode on port 5001 (debug=True, reloader=False)")
            app.run(debug=True, host='0.0.0.0', port=5001, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        cleanup_mqtt()
        sys.exit(0)
