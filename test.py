import joblib

# Load the model
model = joblib.load('svm_model.pkl')

import cv2
import mediapipe as mp
import numpy as np

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def extract_landmarks(image_path, num_landmarks=468):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    
    if not result.multi_face_landmarks:
        return None  # No face detected

    data = []
    for face_landmarks in result.multi_face_landmarks:
        for i in range(1, num_landmarks):
            landmark = face_landmarks.landmark[i - 1]
            data.extend([landmark.x, landmark.y, landmark.z])
        break  # Only the first face
    return np.array(data).reshape(1, -1)

# Example usage
image_path = '/Users/pranaymishra/Desktop/deepfake_detector/image.png'
features = extract_landmarks(image_path)

if features is not None:
    prediction = model.predict(features)
    print('Real' if prediction[0] == 0 else 'Fake')
else:
    print('No face detected in the image')
