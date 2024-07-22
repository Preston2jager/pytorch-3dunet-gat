import streamlit as st
from Data_Points import Page_Datapoints
from Data_Graph import Data_Graph
from Data_Utilities import *

def Page_Datahandler():
    st.title("Data Handler")
    st.write("Point data and graph are all required for training and predicting.")
    Update()
    SUB_PAGES = {
        "Point Data": Page_Datapoints,
        "Graph Data": Data_Graph
    }
    sub_selection = st.radio("选择子页面", list(SUB_PAGES.keys()))
    sub_page = SUB_PAGES[sub_selection]
    sub_page()