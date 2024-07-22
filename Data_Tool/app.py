import torch
import numpy as np
import streamlit as st
from Data_Utilities import *

from Data_Handler import Page_Datahandler

Points_Dir = ""

PAGES = {
    "Data Handler": Page_Datahandler
}

st.title("3D-Unet-GAT Toolset")

st.sidebar.title("Tool list:")
selection = st.sidebar.radio("Proceed from top", list(PAGES.keys()))

page = PAGES[selection]
page()



