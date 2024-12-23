import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Define the custom loss function
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weighted_ce = cross_entropy * weights
        return K.sum(weighted_ce, axis=-1)

    return loss

# Class weights array (update this with your actual class weights)
class_weights_array = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Replace with actual weights

# Load the model
model_path = 'C:/Users/Melvin Tang/OneDrive/Codes/I2ML/FINALPRO/model/cnn_trialNew.h5'
model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights_array), metrics=['accuracy'])

# Define emotion categories
categories = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Define a dictionary mapping emotions to filter file paths
filter_paths = {
    'angry': 'E:/Projects/real-time-emotion-recognition/filters/path_to_angry_filter.png',
    'fear': "E:/Projects/real-time-emotion-recognition/filters/path_to_fear_filter.png",
    'happy': "E:/Projects/real-time-emotion-recognition/filters/path_to_happy_filter.png",
    'neutral': "E:/Projects/real-time-emotion-recognition/filters/path_to_neutral_filter.png",
    'sad': "E:/Projects/real-time-emotion-recognition/filters/path_to_sad_filter.png"
}

# Load filters into a dictionary
filters = {}
for emotion, path in filter_paths.items():
    filters[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Read with alpha channel

# Function to overlay a filter on the face
def overlay_filter(frame, filter_img, x, y, w, h):
    # Get original filter dimensions
    fh, fw = filter_img.shape[:2]
    
    # Calculate the new dimensions while maintaining the aspect ratio
    aspect_ratio = fw / fh
    if w / h > aspect_ratio:  # Face is wider than the filter
        new_h = h
        new_w = int(h * aspect_ratio)
    else:  # Face is taller than the filter
        new_w = w
        new_h = int(w / aspect_ratio)

    # Resize the filter to the new dimensions
    filter_resized = cv2.resize(filter_img, (new_w, new_h))

    # Calculate the position to center the filter on the face
    x_offset = x + (w - new_w) // 2
    y_offset = y + (h - new_h) // 2

    # Split the filter into its color and alpha channels
    filter_rgb = filter_resized[:, :, :3]  # Color channels
    filter_alpha = filter_resized[:, :, 3]  # Alpha channel (transparency)

    # Ensure the filter is within the frame boundaries
    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
    
    # Adjust filter size if it goes out of bounds
    filter_rgb = filter_rgb[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]
    filter_alpha = filter_alpha[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]

    # Extract the region of interest (ROI) from the frame
    roi = frame[y1:y2, x1:x2]

    # Blend the filter with the ROI
    for c in range(3):  # Loop over color channels
        roi[:, :, c] = roi[:, :, c] * (1 - filter_alpha / 255.0) + filter_rgb[:, :, c] * (filter_alpha / 255.0)

    # Place the blended ROI back into the frame
    frame[y1:y2, x1:x2] = roi

# Start the webcam for real-time prediction
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply noise reduction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Noise reduction

    # Detect faces using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        blurred_frame, 
        scaleFactor=1.3,  # Adjusted
        minNeighbors=7,   # Adjusted
        minSize=(50, 50)  # Ignore very small detections
    )

    # Post-detection filtering
    filtered_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.75 < aspect_ratio < 1.3:  # Keep approximately square detections
            filtered_faces.append((x, y, w, h))

    # Process filtered faces
    for (x, y, w, h) in filtered_faces:
        # Extract the face region
        face = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48, 48))  # Assuming input size for model is 48x48
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(48, 48, 1)
        rgb_face = np.concatenate([reshaped_face] * 3, axis=-1)
        rgb_face = np.expand_dims(rgb_face, axis=0)

        # Make a prediction
        predictions = model.predict(rgb_face, verbose=0)
        predicted_class = np.argmax(predictions)
        predicted_emotion = categories[predicted_class]

        # Get the corresponding filter
        filter_img = filters.get(predicted_emotion, None)
        if filter_img is not None:
            # Overlay the filter
            overlay_filter(frame, filter_img, x, y, w, h)

    # Show the video feed
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
