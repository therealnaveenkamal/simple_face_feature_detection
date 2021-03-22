import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def detect(image):
    image = np.array(image.convert('RGB'))
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
    return image, faces


def about():
	st.write(
		'''
        Face Features Detection by Machine Learning Knowledge

        This app is built by using OpenCV library and Streamlit package.
		''')


def main():
    """Face Features Detection App"""
    local_css("style.css")
    st.image("mlk1.png")
    t = "<h2 class='title blue'>Face Features Detection App</h2>"
    st.markdown(t,unsafe_allow_html=True)
    te = "<div class='title blue'>Built with OpenCV and Streamlit</div>"
    st.markdown(te,unsafe_allow_html=True)
    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    if choice == "Home":
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    	if image_file is not None:
    		image = Image.open(image_file)
    		if st.button("Process"):
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Found {} faces\n".format(len(result_faces)))
    elif choice == "About":
    	about()
	
if __name__ == "__main__":
    main()
