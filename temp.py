from ultralytics import solutions
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize video capture
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize AIGym
gym = solutions.AIGym(
    show=True,
    kpts=[5,11, 13, 15],  # Keypoints for squat monitoring
    up_angle=160.0,
    down_angle=70.0,
    model="yolo11n-pose.pt"
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = gym.monitor(im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
BEN BURDA BİR ŞEYLER YAPTIM
cv2.destroyAllWindows()