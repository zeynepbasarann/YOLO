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
gym_squat = solutions.AIGym(
    show=False,  # Show parametresini False yapıyoruz
    kpts=[5,11, 13, 15],  # Squat için keypoints
    up_angle=160.0,
    down_angle=90.0,
    model="yolo11n-pose.pt"
)
# AIGym modelini başlatma - Köprü (Bridge) egzersizini izlemek için özel olarak yapılandırıldı
gym_bridge = solutions.AIGym(
    show=True,                       # Sonuçların ekranda gösterilmesini sağlar
    kpts=[11, 13, 15],               # İzlenecek anahtar noktalar: Kalça, diz, ayak bileği (Bridge için)
    up_angle=145.0,                  # Yukarı pozisyonu tanımlayan açı eşiği (145 derece)
    down_angle=90.0,                 # Aşağı pozisyonu tanımlayan açı eşiği (90 derece)
    model="yolo11n-pose.pt"          # Kullanılacak model dosyası
)

last_time = time.time()
squat_started = False
squat_completed = False
total_squats = 0
bridge_started = False      # Köprü hareketinin başladığını belirtir
bridge_completed = False     # Köprü hareketinin tamamlandığını belirtir
total_bridges = 0            # Toplam tamamlanan köprü hareketi sayısı


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
        
    results = gym_squat.monitor(im0)
    results2 = gym_bridge.monitor(im0)
    if hasattr(gym_squat, 'angle') and gym_squat.angle and len(gym_squat.angle) > 0:
        current_angle = gym_squat.angle[0]
        
        # Squat başlangıcı - yukarıdan aşağıya
        if current_angle < 90 and not squat_started:
            squat_started = True
            squat_completed = False
            
        # Squat tamamlanması - aşağıdan yukarıya
        if squat_started and not squat_completed and current_angle > 150:
            if time.time() - last_time > 1.0:
                total_squats += 1
                last_time = time.time()
                squat_completed = True
                squat_started = False
                
    # Açı değerlerini kontrol et
    if hasattr(gym_bridge, 'angle') and gym_bridge.angle and len(gym_bridge.angle) > 0:
        current_angle = gym_bridge.angle[0]  # Açı değerinin ilk öğesini al (tek bir açı bekleniyor)
        
        # Köprü hareketinin başlangıcı - Yukarı çıkış hareketi
        if current_angle < 90 and not bridge_started:
            bridge_started = True
            bridge_completed = False
            
        # Köprü hareketinin tamamlanması - Yukarıda pozisyonu koruma
        if bridge_started and not bridge_completed and current_angle > 145:
            if time.time() - last_time > 1.0:  # Pozisyonun 1 saniye boyunca tutulup tutulmadığını kontrol et
                total_bridges += 1             # Köprü sayısını artır
                last_time = time.time()        # Zamanlayıcıyı sıfırla
                bridge_completed = True        # Hareket tamamlandı olarak işaretle
                bridge_started = False         # Yeni hareketi beklemeye geç
                
    # Ekrana toplam köprü sayısını yazdır
    cv2.putText(im0, f"Bridge Count: {total_bridges}", (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Görüntüyü ekranda göster
    cv2.imshow("Exercise Monitor", im0)


                
    # Ekranda göster        
    cv2.putText(im0, f"Squat Count: {total_squats}", (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow("Exercise Monitor", im0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

