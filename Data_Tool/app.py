import torch
import numpy as np
import streamlit as st


from Data_Handler import Page_Datahandler
from Data_Points import Page_Datapoints

PAGES = {
    "Data Handler": Page_Datahandler,
    "Data Points": Page_Datapoints
}

st.title("3D-Unet-GAT Toolset")

st.sidebar.title("Tool list:")
selection = st.sidebar.radio("Tool list:", list(PAGES.keys()))

page = PAGES[selection]
page()

# 你可以在此处添加更多功能
