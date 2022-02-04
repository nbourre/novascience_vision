import cv2
import os

# Function that returns the closest width and height ratio divisble by 32, but greater than base_width
def get_closest_ratio_divisble_by_32(frame, base_width = 256):
    y, x, c = frame.shape
    ratio_base_width = base_width / x
    height = round(y * ratio_base_width / 32) * 32
    width = base_width
    
    return height, width

os.chdir('I:/My Drive/_cegep/NovaScience/projet_vision/videos/classees/videoPersonneTombant')

cap = cv2.VideoCapture('1.mp4')

ret, frame = cap.read()

print (frame.shape)

print (get_closest_ratio_divisble_by_32(frame))