import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRgb)
    #print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec,drawSpec)



    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    cv2.imshow("Image",img)
    cv2.waitKey(1)