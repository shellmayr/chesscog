#!/usr/bin/env python3
import os
import ssl
import json
import base64
import binascii
from datetime import datetime
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_URL = os.getenv('MQTT_URL')
MQTT_HOST = os.getenv('MQTT_HOST')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_INSECURE = os.getenv('MQTT_INSECURE', 'false').lower() == 'true'
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/hypervision/forka/device/10:51:DB:85:4B:B0/feed')
MQTT_CLIENT_ID = os.getenv('MQTT_CLIENT_ID', 'chess_mqtt_client')

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        broker_info = MQTT_URL if MQTT_URL else f"{MQTT_HOST}:{MQTT_PORT}"
        print(f"Connected to MQTT broker: {broker_info}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect to MQTT broker. Return code: {rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    message = msg.payload.decode('utf-8')
    print(f"Received message on topic '{topic}': {message}")
    
    try:
        # Parse JSON message
        data = json.loads(message)
        
        # Check if this is an image message
        if data.get("type") == "image" and "image" in data:
            # Extract base64 image data
            b64_image = data["image"]
            
            # Decode base64 image
            image_data = base64.b64decode(b64_image)
            
            # Create images directory if it doesn't exist
            os.makedirs("images", exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"images/image_{timestamp}.jpg"
            
            # Save image as JPEG
            with open(filename, "wb") as f:
                f.write(image_data)
            
            print(f"Saved image as: {filename}")
            
    except json.JSONDecodeError:
        # Not a JSON message, skip image processing
        pass
    except (KeyError, binascii.Error) as e:
        print(f"Error processing image: {e}")
    except Exception as e:
        print(f"Unexpected error handling message: {e}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker")

def main():
    # Determine broker connection details
    if MQTT_URL:
        # Parse URL format (e.g., mqtt://broker.example.com:1883 or mqtts://broker.example.com:8883)
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
        print("Error: Missing required MQTT credentials in .env file")
        print("Please ensure MQTT_HOST or MQTT_URL, MQTT_USERNAME, and MQTT_PASSWORD are set")
        return

    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    if use_ssl:
        if MQTT_INSECURE:
            # Disable SSL certificate verification for self-signed certificates
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        else:
            client.tls_set()
    
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        connection_type = "secure" if use_ssl else "insecure"
        print(f"Connecting to MQTT broker at {broker_host}:{broker_port} ({connection_type})...")
        client.connect(broker_host, broker_port, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.disconnect()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
