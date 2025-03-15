from ultralytics import solutions
import cv2
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Video yakalama
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# AIGym başlatma
gym = solutions.AIGym(
    show=False,  # Show parametresini False yapıyoruz
    kpts=[5,11, 13, 15],  # Squat için keypoints
    up_angle=160.0,
    down_angle=90.0,
    model="yolo11n-pose.pt"
)

last_time = time.time()
squat_started = False
squat_completed = False
total_steps = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
        
    results = gym.monitor(im0)
    
    if hasattr(gym, 'angle') and gym.angle and len(gym.angle) > 0:
        current_angle = gym.angle[0]
        
        # Squat başlangıcı - yukarıdan aşağıya
        if current_angle < 90 and not squat_started:
            squat_started = True
            squat_completed = False
            
        # Squat tamamlanması - aşağıdan yukarıya
        if squat_started and not squat_completed and current_angle > 150:
            if time.time() - last_time > 1.0:
                total_steps += 1
                last_time = time.time()
                squat_completed = True
                squat_started = False
                
    # Ekranda göster        
    cv2.putText(im0, f"Squat Count: {total_steps}", (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow("Exercise Monitor", im0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Toplam Squat Sayısı: {total_steps}")