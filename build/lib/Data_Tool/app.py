import numpy as np
import streamlit as st
from Utilities.Data_Utilities import *

from Pages.Page_Intro import Page_Intro
from Pages.Data_Handler import Page_Datahandler
from Pages.Page_Train import Page_Train

PAGES = {
    "Introduction":Page_Intro,
    "Data Handler":Page_Datahandler,
    "Training":Page_Train,
}

st.sidebar.title("3D-Unet-GAT Toolset")

selection = st.sidebar.radio("Tool list:", list(PAGES.keys()))

page = PAGES[selection]
page()



