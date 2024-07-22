import streamlit as st
from Data_Points import Page_Datapoints
from Data_Graph import Data_Graph

def Page_Datahandler():
    st.title("Data Scriber")
    "3D-U-Net-GAT data handler\n"
    "Chooes your task:\n"
    "1: Display current graph\n"
    "2: Add a node\n"
    "3: Add an edge\n"
    "4: Reset graph\n"
    "5: Export data set\n"
    "6: Check HDF5 file status\n"
    "7: Exit\n"

    SUB_PAGES = {
        "Point Data": Page_Datapoints,
        "Graph Data": Data_Graph
    }
    sub_selection = st.radio("选择子页面", list(SUB_PAGES.keys()))
    sub_page = SUB_PAGES[sub_selection]
    sub_page()