from pyexpat import features
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from keras.losses import CosineSimilarity
import argparse
import cv2

ap =argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True)
args = vars(ap.parse_args())

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


add_face_to_db('./image1.jpg', 'Huy')
# add_face_to_db('./image4.jpg','Nguoi la')
add_face_to_db('./image7.jpg','Tri')
detect_verify = FaceDetection()
feature_extract_verify = FeatureExtraction()
detect_verify.load_img(args['image'])
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

face_img_verify = cv2.imread(args['image'])
cv2.putText(face_img_verify,face_label,(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
cv2.imshow('Output', face_img_verify)
cv2.waitKey(0)