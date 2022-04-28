import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np

model = VGG16(weights='imagenet', include_top=False)


class FeatureExtraction:
    def load_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        img_resize = img_pil.resize((224,224))
        x = image.img_to_array(img_resize)
        self.img = np.expand_dims(x, axis=0)
        
    
    def extract(self):
        x = preprocess_input(self.img)
        features = model.predict(x)
        return features



