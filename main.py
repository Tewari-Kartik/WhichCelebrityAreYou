import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet

# -------------------- PAGE --------------------
st.set_page_config(page_title="Celebrity Twin", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>

/* 🔥 Neon glowing background */
.stApp {
    background: linear-gradient(135deg, #1f005c, #5b0060, #870160, #ac255e, #ca485c);
    background-size: 300% 300%;
    animation: glowBG 8s ease infinite;
}

@keyframes glowBG {
    50% { background-position: 100% 50%; }
}

/* Title */
.title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 0 0 10px #ff00ff, 0 0 20px #ff00ff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    opacity: 0.85;
    margin-bottom: 25px;
}

/* Result text */
.result {
    font-size: 2rem;
    font-weight: bold;
    margin-top: 15px;
    color: #00fff7;
    text-align: center;
    text-shadow: 0 0 10px #00fff7;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff00cc, #3333ff);
    color: white;
    border-radius: 12px;
    padding: 12px;
    font-weight: bold;
    box-shadow: 0 0 15px rgba(255,0,255,0.5);
}

/* Images */
img {
    border-radius: 15px;
    border: 2px solid rgba(255,255,255,0.2);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<div class='title'>🎬 Celebrity Look-Alike Finder</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your photo and discover your Bollywood twin</div>", unsafe_allow_html=True)

# -------------------- MODELS --------------------
detector = MTCNN()
embedder = FaceNet()

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

norms = np.linalg.norm(feature_list, axis=1, keepdims=True)
feature_list = feature_list / np.where(norms == 0, 1, norms)

# -------------------- FUNCTIONS --------------------
def save_uploaded_image(uploaded_image):
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', uploaded_image.name)
    with open(path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    return path

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)

    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160,160))
    face = np.expand_dims(face.astype('float32'), axis=0)

    embedding = embedder.embeddings(face)[0]
    norm = np.linalg.norm(embedding)

    return embedding / (norm if norm != 0 else 1)

def recommend(features):
    similarity = cosine_similarity([features], feature_list)[0]
    idx = np.argmax(similarity)
    return idx, similarity

# -------------------- UPLOAD --------------------
uploaded_image = st.file_uploader("📸 Upload your image", type=["jpg","png","jpeg"])

# -------------------- MAIN --------------------
if uploaded_image is not None:

    path = save_uploaded_image(uploaded_image)
    user_img = Image.open(uploaded_image)

    with st.spinner("✨ Matching your face..."):
        features = extract_features(path)

    if features is not None:
        idx, similarity = recommend(features)

        actor = os.path.basename(filenames[idx]).split('.')[0].replace('_',' ')
        score = similarity[idx]

        st.divider()

        # 🔥 MEDIUM SIZE IMAGES
        col1, col2 = st.columns(2)

        with col1:
            st.image(user_img, caption="Your Image", width=350)

        with col2:
            st.image(filenames[idx], caption="Celebrity Match", width=350)

        # RESULT TEXT
        if score > 0.75:
            msg = f"🔥 Damn! You ARE {actor}"
        elif score > 0.55:
            msg = f"😍 Wow! You look like {actor}"
        else:
            msg = f"😅 Slight resemblance to {actor}"

        st.markdown(f"<div class='result'>{msg}</div>", unsafe_allow_html=True)

        # PROGRESS
        st.progress(float(score))
        st.caption(f"Confidence: {score:.2f}")

        st.divider()

        if st.button("🔄 Try Another Image"):
            st.rerun()

    else:
        st.error("❌ No face detected. Try a clearer image.")

# -------------------- FOOTER --------------------
st.markdown("<div class='footer'>Made with ❤️ by Kartik 🚀</div>", unsafe_allow_html=True)