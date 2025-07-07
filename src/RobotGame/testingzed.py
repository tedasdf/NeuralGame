# import pyzed.sl as sl
# import cv2
# import numpy as np
# import time 


# zed = sl.Camera()

# devices = zed.get_device_list()
# for i, dev in enumerate(devices):
#     print(f"Device {i}: {dev.serial_number} - {dev.model} - {dev.camera_model}")

# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD720
# init_params.camera_fps = 30

# input_type = sl.InputType()
# input_type.set_from_serial_number(devices[0].serial_number)

# init_params.input = input_type

# status = zed.open(init_params)
# if status != sl.ERROR_CODE.SUCCESS:
#     print(f"Error opening camera: {status}")
#     exit(1)

# print("Camera opened successfully.")

# left_image = sl.Mat()
# right_image = sl.Mat()
# print("Starting camera capture...")

# while True:
#     if zed.grab() == sl.ERROR_CODE.SUCCESS:
#         zed.retrieve_image(left_image, sl.VIEW.LEFT)
#         zed.retrieve_image(right_image, sl.VIEW.RIGHT)

#         left_image_np = left_image.get_data()
#         right_image_np = right_image.get_data()

#         cv2.imshow("Left Image", left_image_np)
#         cv2.imshow("Right Image", right_image_np)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print("Failed to grab image.")
# cv2.destroyAllWindows()
# zed.close()
# # This code initializes a ZED camera, retrieves images from both left and right cameras, and displays them in real-time.

import cv2

cap = cv2.VideoCapture("rtsp://192.168.0.102:554/mjpeg/1")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("ESP32-CAM", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()