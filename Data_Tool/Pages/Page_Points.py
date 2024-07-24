import streamlit as st
import tempfile
import subprocess
from Data_Utilities import *
from Global import *

def Page_DataPoints():
    st.title("Point Data ")
    uploaded_file = st.file_uploader("Upload the point data file in npy format")
    if uploaded_file is not None:
        st.write("File loaded")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            Points_Dir = tmpfile.name
        st.write('File saved at:', Points_Dir)
        if st.button('Render'):
            subprocess.run(['python', './Data_Render.py', '--file', Points_Dir], check=True)
    
    