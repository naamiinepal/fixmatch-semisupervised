import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import BasicNet
from dataset import data_transform,n_classes
import medmnist
from medmnist import INFO
import numpy as np

data_flag = 'pathmnist'

st.header('PathMNIST')
def predict(image):
    classifier_model = 'checkpoint/2024-01-16 14:23:51/best_model_8_accuracy=0.8735.pt'
    model = BasicNet(in_channels=3,num_classes=n_classes) 
    model.load_state_dict(torch.load(classifier_model,map_location='cpu'))
    model.eval()
    image_tensor = data_transform(image)
    out_logit = model(image_tensor.unsqueeze(0))
    scores = nn.Softmax(dim=1)(out_logit).squeeze(0)
    prediction = torch.argmax(scores).item()
    return INFO[data_flag]['label'][str(prediction)],torch.max(scores).item()

def main():
    file_uploaded = st.file_uploader('Choose File',type=['png','jpg','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        label,score = predict(image)
        st.write({'prediction':label,'score':score})
        st.pyplot(fig)
    
if __name__ == '__main__':
    main()