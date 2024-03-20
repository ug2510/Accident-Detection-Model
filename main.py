 import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import time

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('1.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
accident_frames = 0
severe_threshold = 11 # Define threshold for severe accident
moderate_threshold = 9  # Define threshold for moderate accident

# Define scale parameters
pixels_per_meter = 100  # Adjust this value according to your specific scenario

# Previous frame data for speed calculation
prev_frame_time = None
prev_car_positions = {}
prev_speeds = {}

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    curr_car_positions = {}  # To store positions of cars in current frame
    curr_speeds = {}  # To store speeds of cars in current frame

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#         cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        # Store car positions for speed calculation
        curr_car_positions[index] = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Calculate speed for each car
        if prev_frame_time is not None:
            time_diff = time.time() - prev_frame_time
            if index in prev_car_positions:
                prev_x, prev_y = prev_car_positions[index]
                distance_traveled = abs((x1 + x2) / 2 - prev_x)
                speed_pixels_per_second = distance_traveled / time_diff

                # Convert speed to meters per second
                speed_meters_per_second = speed_pixels_per_second / pixels_per_meter

                # Store speed for each car
                curr_speeds[index] = speed_meters_per_second

                # Display speed on bounding box
                speed_text = f"Speed: {speed_meters_per_second:.2f} m/s"
#                 cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Update previous frame data
    prev_frame_time = time.time()
    prev_car_positions = curr_car_positions

    # Check for accidents based on deceleration
    for index, speed in curr_speeds.items():
        if index in prev_speeds:
            prev_speed = prev_speeds[index]
            deceleration = prev_speed - speed
            if deceleration > 1:  # Adjust threshold as needed
                accident_frames += 1

    # Classify accident severity based on the number of frames overlapping
    if accident_frames >= severe_threshold:
        severity = "Severe Accident"
    elif accident_frames >= moderate_threshold:
        severity = "Moderate Accident"
    else:
        severity = " "

    cv2.putText(frame, severity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Update previous speeds
    prev_speeds = curr_speeds

cap.release()
cv2.destroyAllWindows()
