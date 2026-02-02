from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import json
import os

app = Flask(__name__)

# File to store memory
MEMORY_FILE = "alya_face_memory.json"

# Load known faces from file
known_face_encodings = []
known_face_names = []

def load_memory():
    global known_face_encodings, known_face_names
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
            known_face_names = data['names']
            # Convert list back to numpy arrays
            known_face_encodings = [np.array(e) for e in data['encodings']]
            print(f"Dr. Alya loaded {len(known_face_names)} faces from memory.")

def save_memory():
    data = {
        'names': known_face_names,
        'encodings': [e.tolist() for e in known_face_encodings] # Convert numpy to list for JSON
    }
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f)

load_memory()

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        image = decode_image(data['image'])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        found_name = "Unknown"

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            
            if True in matches:
                first_match_index = matches.index(True)
                found_name = known_face_names[first_match_index]
                break # Just find the first person
        
        return jsonify({"name": found_name})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        name = data['name']
        image = decode_image(data['image'])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)
            save_memory()
            return jsonify({"status": "success", "message": f"Added {name} to memory."})
        else:
            return jsonify({"status": "failed", "message": "No face detected to register."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
