import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Optional if using a GPU
# Prevent the code to consume all the VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
  for person in keypoints_with_scores:
    draw_connections(frame, person, edges, confidence_threshold)
    draw_keypoints(frame, person, confidence_threshold)

# Function that returns the closest width and height ratio divisble by 32, but greater than base_width
def get_closest_ratio_divisble_by_32(frame, base_width = 256):
    y, x, c = frame.shape
    ratio_base_width = base_width / x
    height = round(y * ratio_base_width / 32) * 32
    width = base_width
    
    return height, width


os.chdir('I:/My Drive/_cegep/NovaScience/projet_vision/videos/classees/videoPersonneTombant')

test_file = '2.mp4'

cap = cv2.VideoCapture(test_file)

resize_y, resize_x = get_closest_ratio_divisble_by_32(cap.read()[1])

confidence_threshold = 0.3

while True:
  ret, frame = cap.read()

  if not ret:
    break

  # resize image
  img = frame.copy()
  img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), resize_y, resize_x)
  input_img = tf.cast(img, dtype=tf.int32)

  # Detection section
  results = movenet(input_img)
  keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6,17,3))

  loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)

  cv2.imshow('output', frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()