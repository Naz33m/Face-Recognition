# Face Recognition API

A tiny face registry over HTTP: send a photo with a name to remember someone, send another photo to ask who it is. Built to learn the basics behind larger real-time camera pipelines.

> Privacy: face encodings are stored in a local JSON file. Only register people who agreed to it, and `alya_face_memory.json` is git-ignored.


A Flask-based face recognition API that can register and recognize faces using deep learning.

## Features

- **Register faces**: Add new faces to the system's memory
- **Recognize faces**: Identify registered faces from images
- **Persistent memory**: Face data is saved to a JSON file for future sessions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The `face_recognition` library requires `dlib`, which may need additional setup:
- **Windows**: You may need to install CMake and Visual Studio Build Tools
- **Linux**: Install `cmake` and `python3-dev`
- **macOS**: Should work out of the box with Xcode Command Line Tools

## Usage

### Start the Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5001`

### API Endpoints

#### 1. Register a Face

**Endpoint**: `POST /register`

**Request Body**:
```json
{
  "name": "John Doe",
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Added John Doe to memory."
}
```

#### 2. Recognize a Face

**Endpoint**: `POST /recognize`

**Request Body**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "name": "John Doe"
}
```

If no match is found, the response will be:
```json
{
  "name": "Unknown"
}
```

## How It Works

1. **Registration**: When you register a face, the system:
   - Decodes the base64 image
   - Detects faces in the image
   - Generates face encodings (128-dimensional vectors)
   - Saves the encoding and name to `alya_face_memory.json`

2. **Recognition**: When you recognize a face, the system:
   - Decodes the base64 image
   - Detects faces in the image
   - Compares face encodings with known faces (tolerance: 0.5)
   - Returns the name of the matching face or "Unknown"

## Data Storage

Face encodings are stored in `alya_face_memory.json` with the following structure:
```json
{
  "names": ["John Doe", "Jane Smith"],
  "encodings": [[...], [...]]
}
```

## Example Usage with Python

```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Register a face
response = requests.post('http://localhost:5001/register', json={
    'name': 'John Doe',
    'image': image_data
})
print(response.json())

# Recognize a face
response = requests.post('http://localhost:5001/recognize', json={
    'image': image_data
})
print(response.json())
```

## Technologies Used

- **Flask**: Web framework
- **face_recognition**: Face detection and recognition (built on dlib)
- **OpenCV**: Image processing
- **NumPy**: Numerical operations

## License

MIT License
