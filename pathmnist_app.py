import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
from model import BasicNet

st.header('PathMNIST')
def predict(image):
    classifier_model = 'checkpoint/2024-01-16 12:10:16/best_model_10_accuracy=0.8263.pt'
    model = torch.load(classifier_model,map_location='cpu')


def main():
    file_uploaded = st.file_uploader('Choose File',type=['png','jpg','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(fig)
    
if __name__ == '__main__':
    main()