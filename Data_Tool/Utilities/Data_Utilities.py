import streamlit as st
import os
from streamlit_modal import Modal
from Utilities.Global import *

def Check_Points(Points_Dir):
    if Points_Dir != "":
        Points_status = True
    else:
        Points_status = False
    return Points_status

def Check_Graph():
    Graph_status = True
    return Graph_status

def Update_Points():
    global Points_Dir
    r = Check_Points(Points_Dir)
    if r:
        st.markdown(
            '<p style="color:green;">Point Data Ready</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:red;">Points Data Not Ready</p>',
            unsafe_allow_html=True
        )
    
def Update_Graph():
    r = Check_Graph()
    if r:
        st.markdown(
            '<p style="color:green;">Graph Data Ready</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:red;">Graph Data Not Ready</p>',
            unsafe_allow_html=True
        )

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('选择一个文件', filenames)
    return os.path.join(folder_path, selected_filename)

def Data_Evaluation(valid_percentage):
    if valid_percentage > 50:
        return "Great data"
    elif valid_percentage > 30:
        return "OK data"
    elif valid_percentage >10:
        return "Not Ideal"
    else:
        return "Bad data"
