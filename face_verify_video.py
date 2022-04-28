from traceback import print_tb
import cv2
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from keras.losses import CosineSimilarity

cap = cv2.VideoCapture(0)

faces_data = []

def add_face_to_db(img_path, label):
    detector = FaceDetection()    
    fex = FeatureExtraction()
    detector.load_img(img_path)
    face = detector.detect()
    fex.load_img(face)
    features = fex.extract()
    data = {'features': features, 'label':label}
    faces_data.append(data)

cosine_loss = CosineSimilarity()


add_face_to_db('./image5.jpg', 'Huy')
add_face_to_db('./image8.jpg','Tri')
add_face_to_db('./image10.jpg','Tin')
add_face_to_db('./image10.jpg','Toan')

while True:
    suc,frame = cap.read()
    detect_verify = FaceDetection()
    feature_extract_verify = FeatureExtraction()
    detect_verify.load_frame(frame)
    face_verify = detect_verify.detect()
    feature_extract_verify.load_img(face_verify)
    features_verify = feature_extract_verify.extract()

        
    scores = []
    for f in faces_data:
        distance = - cosine_loss(features_verify, f['features']).numpy()
        scores.append(distance)

    max_score = max(scores)
    max_index = scores.index(max_score)
    face_label = faces_data[max_index]['label']

    # cv2.putText(frame,face_label,(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    print(face_label ,'-',scores)
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()