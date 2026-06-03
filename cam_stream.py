import cv2

DROID_CAM_URL = "http://192.168.1.54/video"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

print("Video stream opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame Dropped, retrying...")
        continue

    cv2.imshow("Droid Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()