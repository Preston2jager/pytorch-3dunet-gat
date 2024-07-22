import streamlit as st
import os
from streamlit_modal import Modal

def Check_Data():
    Points_status = True
    Graph_status = False
    return Points_status, Graph_status

def Update():
    r1,r2 = Check_Data()
    if r1:
        st.write("Point Data Ready")
    else:
        st.markdown(
            '<p style="color:red;">Points Data Not Ready</p>',
            unsafe_allow_html=True
        )
    if r2:
        st.write("Graph Data ready")
    else:
        st.markdown(
            '<p style="color:red;">Graph Data Not Ready</p>',
            unsafe_allow_html=True
        )

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('选择一个文件', filenames)
    return os.path.join(folder_path, selected_filename)
