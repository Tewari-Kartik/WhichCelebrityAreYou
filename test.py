import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

# Load data
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# safe normalization
norms = np.linalg.norm(feature_list, axis=1, keepdims=True)
feature_list = feature_list / np.where(norms == 0, 1, norms)

detector = MTCNN()
embedder = FaceNet()

img_path = 'sample/satya.png'  # change if needed
img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(img)

if len(results) == 0:
    print("No face detected")
    exit()

x, y, w, h = results[0]['box']
x, y = abs(x), abs(y)

face = img[y:y+h, x:x+w]
face = cv2.resize(face, (160,160))

face = face.astype('float32')
face = np.expand_dims(face, axis=0)

embedding = embedder.embeddings(face)[0]

# safe normalization
norm = np.linalg.norm(embedding)
embedding = embedding / (norm if norm != 0 else 1)

similarity = cosine_similarity([embedding], feature_list)[0]

top_5 = np.argsort(similarity)[-5:][::-1]

print("\nTop Matches:")
for idx in top_5:
    actor = os.path.basename(filenames[idx]) \
                .split('.')[0].replace('_',' ')
    print(actor, "Score:", round(similarity[idx], 2))