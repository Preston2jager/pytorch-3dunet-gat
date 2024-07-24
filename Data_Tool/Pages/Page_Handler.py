import streamlit as st
from Pages.Page_Points import Page_DataPoints
from Pages.Page_Graph import Page_DataGraph
from Utilities.Data_Utilities import *

def Page_Datahandler():
    st.title("Data Handler")
    st.write("Point data and graph are all required for training and predicting.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h1 style="font-size: 24px;">Point Data status</h1>', unsafe_allow_html=True)
        Update_Points()
    with col2:
        st.markdown('<h1 style="font-size: 24px;">Graph Data status</h1>', unsafe_allow_html=True)
        Update_Graph()

    SUB_PAGES = {
        "Point Data": Page_DataPoints,
        "Graph Data": Page_DataGraph
    }
    sub_selection = st.radio("选择子页面", list(SUB_PAGES.keys()))
    sub_page = SUB_PAGES[sub_selection]
    sub_page()