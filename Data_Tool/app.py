import torch
import numpy as np
import streamlit as st
from Utilities.Data_Utilities import *

from Pages.Page_Howto import Page_Howto
from Pages.Data_Handler import Page_Datahandler


PAGES = {
    "Introduction":Page_Howto,
    "Data Handler":Page_Datahandler
}

st.sidebar.title("3D-Unet-GAT Toolset")

selection = st.sidebar.radio("Tool list:", list(PAGES.keys()))

page = PAGES[selection]
page()



