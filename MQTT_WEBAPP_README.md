# MQTT Live Message Viewer

A web application that subscribes to MQTT topics and displays messages in real-time with auto-refresh functionality.

## Features

- 🔴 **Real-time message display** using Server-Sent Events (SSE)
- 📡 **MQTT integration** using the existing mqtt_subscriber.py configuration
- 🖼️ **Image message support** with automatic base64 decoding and display
- 🔍 **Chess Recognition Pipeline** - Full integration with ChessCog recognition system
- 📊 **JSON message formatting** with syntax highlighting
- 🔄 **Auto-scroll functionality** with toggle control
- 📈 **Connection status monitoring** and message statistics
- 🧹 **Clean UI** with message categorization and timestamps
- 🎯 **Multiple Image Views** - Original, corner detection, and board analysis
- ♟️ **Live Chess Analysis** - Board state, FEN notation, and piece detection

## Files Created

1. `mqtt_live_webapp.py` - Main web application
2. `mqtt_test_publisher.py` - Test publisher for sending sample messages
3. `MQTT_WEBAPP_README.md` - This documentation

## Setup

The webapp uses the same MQTT configuration as your existing `mqtt_subscriber.py`, so ensure your `.env` file contains:

```env
MQTT_URL=your_mqtt_broker_url
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password  
MQTT_TOPIC=your_topic
# ... other MQTT settings
```

## Usage

### Running the Webapp

1. **Development Mode** (recommended for testing):
   ```bash
   python mqtt_live_webapp.py
   ```

2. **Production Mode** (more stable, no debug features):
   ```bash
   python mqtt_live_webapp.py --production
   # or set FLASK_ENV=production
   ```

The webapp will start on `http://localhost:5001`

## Chess Recognition Setup

The webapp includes full integration with the ChessCog chess recognition pipeline. If the recognition modules are available, images will be automatically processed to detect chess boards, corners, and pieces.

### Requirements for Chess Recognition

The webapp will automatically detect if chess recognition is available. For full functionality, ensure:

1. **ChessCog modules** are properly installed and accessible
2. **Model files** are available in the `models://` directory
3. **GPU support** (optional but recommended for better performance)

If chess recognition is not available, the webapp will still function normally but only display original images without analysis.

### Recognition Features

When an image is received via MQTT:

1. **Automatic Processing**: Image is immediately analyzed for chess content
2. **Corner Detection**: Identifies chess board boundaries and perspective
3. **Piece Recognition**: Detects and classifies individual chess pieces
4. **Board State**: Generates FEN notation and board visualization
5. **Multiple Views**: Displays original, corner detection, and board state
6. **Robust Error Handling**: Graceful fallback when no chess board is detected
7. **Smart Detection**: Distinguishes between "no board found" vs "processing errors"

### Testing with Sample Messages

Run the test publisher to send sample messages:

```bash
python mqtt_test_publisher.py
```

This will send various test messages including:
- Simple text messages
- JSON messages with structured data
- Chess-related messages
- Image messages (small test images)
- Periodic status updates

## UI Features

- **Dual-Panel Layout**: 
  - Left panel (1/3): Clean activity feed with user-friendly messages
  - Right panel (2/3): Large display of the latest received image
- **Modern shadcn-style Design**: Professional, clean interface with subtle shadows and modern typography
- **Status Bar**: Shows MQTT connection status and statistics with styled indicators
- **Live Activity Feed**: Real-time feed with SVG icons and clean message formatting
- **Large Image Display**: Latest image shown as large as possible in right panel with blur overlay info
- **Smart Message Parsing**: Raw MQTT messages converted to user-friendly activities:
  - 📷 **Image Received** - Shows filename only (image displays large in right panel)
  - ▶️ **Game Started** - Chess game initiation
  - ⏹️ **Game Ended** - Game completion with winner info
  - ♟️ **Move Played** - Chess moves
  - ℹ️ **Status Update** - System status messages
  - ⚠️ **Alerts/Errors** - Connection issues and errors
  - 💬 **Messages** - General data
- **No Technical Details**: MQTT topics, JSON data, and technical details hidden from user
- **Auto-scroll Toggle**: Button in feed header to enable/disable automatic scrolling
- **Message History**: Keeps last 100 messages in memory
- **Chess Recognition Integration**: 
  - 🔍 **Automatic Analysis** - Images automatically processed through ChessCog pipeline
  - 📷 **Multiple Views** - Original image, corner detection, and board state visualization
  - ♟️ **Board Information** - Live FEN notation, piece count, and analysis results
  - 🎯 **Tab Interface** - Easy switching between different analysis views (Board view is default)
  - ✅ **Status Indicators** - Success/failure indicators with error details
  - 🔍 **No Board Detection** - Elegant handling when no chess board is in the image
- **Glass-morphism Overlays**: Modern blur effects for image metadata and board info
- **Smart UI States**: 
  - 🏁 **Board View Default** - Opens directly to board analysis results
  - ⚠️ **No Board Messages** - Clear visual feedback when no chess board is detected
  - 📷 **Overlay Indicators** - Visual status overlays on original images
  - 🎯 **Context-Aware Tabs** - Different behavior per tab when no board is found

## API Endpoints

- `GET /` - Main webapp interface
- `GET /events` - Server-Sent Events stream for real-time updates
- `GET /status` - JSON endpoint with connection status and statistics
- `GET /messages` - JSON endpoint with recent message history

## Troubleshooting

### MQTT Reconnection Issues

The webapp has been configured to minimize reconnection issues:
- Uses `use_reloader=False` in development mode
- Implements connection locking and cleanup
- Conservative keepalive and reconnection settings
- Proper cleanup on shutdown

If you still experience frequent reconnections:
1. Try production mode: `python mqtt_live_webapp.py --production`
2. Check your network connection stability
3. Verify MQTT broker settings and limits

### Connection Problems

- Ensure your `.env` file has correct MQTT credentials
- Check that the MQTT broker is accessible
- Verify SSL/TLS settings match your broker configuration

## Image Message Format

For image messages, send JSON in this format:
```json
{
  "type": "image",
  "image": "base64_encoded_image_data",
  "timestamp": "2025-01-21T10:00:00",
  "description": "Optional description"
}
```

The webapp will automatically decode and display the image.

## Development

The webapp is built with:
- Flask for the web framework
- Server-Sent Events for real-time communication
- paho-mqtt for MQTT client functionality
- Threading for concurrent MQTT and web operations

To extend functionality:
1. Modify message processing in `MQTTHandler.on_message()`
2. Update the HTML template for UI changes
3. Add new API endpoints as needed

## Performance

- Message queue limited to 100 messages to prevent memory issues
- Images are automatically saved to `images/` directory
- Disconnected SSE clients are automatically cleaned up
- MQTT client uses clean sessions and proper resource cleanup
