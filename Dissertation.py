import os
import cv2 # image processing
import numpy as np # numerical operations
import tensorflow as tf # deep learning
import time # timing

from object_detection.utils import label_map_util # loading label maps
from object_detection.utils import visualization_utils as vis_util # drawing detection

# Define paths
MODEL_PATH = 'ssd_mobilenet_v2_fpnlite_320x320/saved_model' # file location of trained object detection model (will be changed)
LABELS_PATH = 'mscoco_label_map.pbtxt'# points to label map file which defines label classes (will be changed)
VIDEO_PATH = 'video.mp4' # path for video file for processing

# Load the TensorFlow model
print("Loading model...")
detection_model = tf.saved_model.load(MODEL_PATH)
print("Model loaded successfully.")

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True) # converts label map into a dictionary for class lookup during detection

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH) # frame by frame processing


# Define preprocessing function
def preprocess_image(image):
    """Enhances defect visibility by applying preprocessing steps."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection using Canny
    return edges


# Process video frames
while cap.isOpened():
    ret, frame = cap.read() #Reads frames from the video. If no frame is read, the loop exits.
    if not ret:
        break

    processed_frame = preprocess_image(frame) #Enhances defect visibility by preprocessing the frame

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...] # adds a new dimension to match the model's input shape

    # Perform object detection
    start_time = time.time()
    detections = detection_model(input_tensor)
    end_time = time.time()

    print(f"Inference time: {end_time - start_time:.2f} seconds")

    # Extract detection results
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()
# Boxes define detected object locations, classes indicate detected object types, and scores represent confidence levels.

    # Visualise results
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5,
        line_thickness=2
    )

    # Display processed frame
    cv2.imshow('Defect Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # loop continues until user presses q
        break

# Cleanup
cap.release() # closes video file
cv2.destroyAllWindows() # closes all OpenCV windows