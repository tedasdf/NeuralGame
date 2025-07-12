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
import requests
import numpy as np
import cv2
import time

def camera_cap_snapshot(self):
    try:
        start_time = time.time()
        response = requests.get("http://192.168.0.102/snapshot", stream=True, timeout=3)
        
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            time_taken = time.time() - start_time
            
            if img is not None:
                return img, time_taken
            else:
                print("Failed to decode image.")
                return None, 0
        else:
            print(f"HTTP error: {response.status_code}")
            return None, 0
    except Exception as e:
        print(f"Exception in snapshot capture: {e}")
        return None, 0


camera_cap_snapshot()