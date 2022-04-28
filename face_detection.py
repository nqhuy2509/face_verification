from mtcnn.mtcnn import  MTCNN
import cv2

# img = cv2.imread('./image9.jpg')
# img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = MTCNN()
# box = detector.detect_faces(img_bgr)[0]
# (x,y,w,h) = box['box']
# face = img[y:y+h, x:x+w]
# cv2.imwrite('face4.jpg', face)
# img = cv2.rectangle(img, (x,y), (x+w, y+h), 255, 2)
# cv2.imshow('Original image', img)
# cv2.imshow('Face', face)
# cv2.waitKey(0)


class FaceDetection:
    def load_frame(self, frame):
        self.img = frame
    def load_img(self, img_path):
        self.img = cv2.imread(img_path)

    def detect(self):
        img_bgr = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        box = detector.detect_faces(img_bgr)[0]
        (x,y,w,h) = box['box']
        return self.img[y:y+h, x:x+w]

    




