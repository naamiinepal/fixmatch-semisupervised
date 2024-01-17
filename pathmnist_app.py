import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
from model import BasicNet
from dataset import data_transform
import medmnist
from medmnist import INFO

data_flag = 'pathmnist'

st.header('PathMNIST')
def predict(image):
    classifier_model = 'checkpoint/2024-01-16 14:23:51/best_model_8_accuracy=0.8735.pt'
    model = torch.load(classifier_model,map_location='cpu')
    image_tensor = data_transform(image.unsqueeze(1))
    print(image_tensor.shape)
    out_logit = model(image_tensor)
    scores = torch.nn.softmax(out_logit)
    confidences = { label:proba for label,proba in zip(scores,INFO[data_flag]['label'])}
    return confidences

def main():
    file_uploaded = st.file_uploader('Choose File',type=['png','jpg','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        predictions = predict(image)
        st.write(predictions)
        st.pyplot(fig)
    
if __name__ == '__main__':
    main()