import cv2
import mediapipe as mp

class FaceMesh():
    def __init__(self, mode=False, maxFace = 2, refineLms=False, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxFace = maxFace
        self.refineLms = refineLms
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1)

    def makeFaceMesh(self, img):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRgb)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms,self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,self.drawSpec)
        return img

def main():
    cap = cv2.VideoCapture(1)
    maker = FaceMesh()
    while True:
        success, img = cap.read()
        img = maker.makeFaceMesh(img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()