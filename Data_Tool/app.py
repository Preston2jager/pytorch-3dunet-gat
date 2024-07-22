import torch
import numpy as np
import streamlit as st
#from Data_Utilities import *

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

graph_data = torch.empty(0, 4)
edge_index = torch.empty(0, 2,dtype=torch.long)



# 你可以在此处添加更多功能
