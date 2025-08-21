#!/usr/bin/env python3
"""
MQTT Test Publisher
Simple script to publish test messages to the MQTT topic for testing the live webapp.
"""

import os
import ssl
import json
import time
import base64
from datetime import datetime
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# MQTT Configuration (same as the webapp)
MQTT_URL = os.getenv('MQTT_URL')
MQTT_HOST = os.getenv('MQTT_HOST') 
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_INSECURE = os.getenv('MQTT_INSECURE', 'false').lower() == 'true'
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/hypervision/forka/device/10:51:DB:85:4B:B0/feed')
MQTT_CLIENT_ID = 'chess_mqtt_test_publisher'

def create_test_messages():
    """Create various test messages."""
    timestamp = datetime.now().isoformat()
    
    messages = [
        # Simple text message
        "Hello from MQTT test publisher!",
        
        # JSON message
        json.dumps({
            "timestamp": timestamp,
            "message": "Test JSON message",
            "sensor_data": {
                "temperature": 23.5,
                "humidity": 45.2,
                "light": 850
            }
        }),
        
        # Chess-related JSON message
        json.dumps({
            "type": "chess_move",
            "timestamp": timestamp,
            "move": "e4",
            "player": "white",
            "game_id": "test_game_001"
        }),
        
        # Status message
        json.dumps({
            "type": "status",
            "timestamp": timestamp,
            "status": "System online",
            "uptime": "2 hours 15 minutes"
        })
    ]
    
    return messages

def create_test_image_message():
    """Create a test image message (small placeholder image)."""
    # Create a small test image (1x1 pixel PNG) as base64
    # This is a tiny red pixel PNG encoded as base64
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFBA8Lq1AAAAABJRU5ErkJggg=="
    
    return json.dumps({
        "type": "image", 
        "timestamp": datetime.now().isoformat(),
        "image": tiny_png_b64,
        "description": "Test image message"
    })

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to MQTT broker for publishing")
    else:
        print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")

def on_publish(client, userdata, mid):
    print(f"📤 Message {mid} published successfully")

def main():
    # Determine broker connection details (same logic as webapp)
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
        print("❌ Error: Missing required MQTT credentials in .env file")
        return

    print(f"🔄 MQTT Test Publisher")
    print(f"📡 Topic: {MQTT_TOPIC}")
    print(f"🌐 Broker: {broker_host}:{broker_port} ({'secure' if use_ssl else 'insecure'})")

    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    if use_ssl:
        if MQTT_INSECURE:
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        else:
            client.tls_set()
    
    client.on_connect = on_connect
    client.on_publish = on_publish

    try:
        print(f"🔌 Connecting to MQTT broker...")
        client.connect(broker_host, broker_port, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        print(f"📤 Publishing test messages...")
        
        # Send regular test messages
        test_messages = create_test_messages()
        for i, message in enumerate(test_messages, 1):
            print(f"📤 Sending message {i}/{len(test_messages)}: {message[:50]}{'...' if len(message) > 50 else ''}")
            result = client.publish(MQTT_TOPIC, message)
            time.sleep(1)  # Wait 1 second between messages
        
        # Send test image message
        print(f"📤 Sending test image message...")
        image_message = create_test_image_message()
        client.publish(MQTT_TOPIC, image_message)
        time.sleep(1)
        
        # Send periodic status updates
        print(f"📤 Sending periodic status updates (press Ctrl+C to stop)...")
        counter = 1
        while True:
            status_msg = json.dumps({
                "type": "periodic_status",
                "timestamp": datetime.now().isoformat(),
                "counter": counter,
                "message": f"Periodic update #{counter}"
            })
            
            print(f"📤 Status update #{counter}")
            client.publish(MQTT_TOPIC, status_msg)
            counter += 1
            time.sleep(10)  # Send status every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping publisher...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"🔌 Disconnected from MQTT broker")

if __name__ == "__main__":
    main()