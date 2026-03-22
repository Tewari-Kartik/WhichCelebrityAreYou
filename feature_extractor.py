import numpy as np
import pickle
from tqdm import tqdm
from keras_facenet import FaceNet
import cv2
from mtcnn import MTCNN

filenames = pickle.load(open('filenames.pkl','rb'))

detector = MTCNN()
embedder = FaceNet()

features = []
valid_filenames = []

for file in tqdm(filenames):
    try:
        img = cv2.imread(file)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img)

        if len(results) == 0:
            continue   # skip images without face

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

        features.append(embedding)
        valid_filenames.append(file)

    except:
        continue

features = np.array(features)

pickle.dump(features, open('embedding.pkl','wb'))
pickle.dump(valid_filenames, open('filenames.pkl','wb'))

print("Embeddings generated successfully!")
print("Total valid images:", len(valid_filenames))