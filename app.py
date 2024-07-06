import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# loading ML model
with open("ECG.pkl", 'rb') as f:
    ecg = pickle.load(f)
heart_diseases = {0:'Normal beat',
                  1:'Supraventricular premature beat',
                  2:'Premature ventricular contraction',
                  3:"Fusion of ventricular and normal beat",
                  4:'Unclassifiable beat'
                 }

st.image("cropedLogo.png")
st.title("I-Care")
st.info("Electrocardiography ECG Scan - Easy Healthcare for Anyone Anytime")

uploaded_file = st.file_uploader("Choose an image...", type=["csv", "xlsx"])

if uploaded_file is not None:
    # if uploaded_file.e
    d = pd.read_csv(uploaded_file)
    st.write(d)
    if d.isnull().sum().sum() !=0:
        d.fillna(d.mean())
    d = d.iloc[0,:].values.reshape(1,-1)    
    # d = d.values.reshape(1,-1)
    output = int(ecg.predict(d)[0])
    disease = heart_diseases[output]
    st.write(f"{disease} detected!")
    st.write("info")
    plt.grid()
    plt.plot(d, label=f"{disease}", c='firebrick')
    plt.suptitle("Electrocardiography ECG")
    plt.legend(loc='upper right')
    plt.show()
