import subprocess
import streamlit as st
from Data_Tool.Utilities.Data_meta import Data_meta

def Page_Train():
    st.title("Train Model")
    data_meta = Data_meta.get_instance()
    data_meta.save_to_json('data_meta.json')
    if st.button('Start Training'):
        command = ['train3dunet', '--config', '../resources/3DUnet_multiclass/train_config.yaml']
        cmd = 'start cmd /c ' + ' '.join(command)
        result = subprocess.Popen(cmd,  shell=True)
        st.text("Training started in a new command window...")

