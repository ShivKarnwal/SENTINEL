import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened!")
    exit()

print("SENTINEL WEBCAM CONNECTED")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Cannot capture frame")
        break

    cv2.imshow("SENTINEL - Live Plant Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
