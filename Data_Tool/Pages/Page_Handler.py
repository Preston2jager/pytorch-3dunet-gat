import streamlit as st
from Pages.Page_Points import Page_DataPoints
from Pages.Page_Graph import Page_DataGraph
from Utilities.Data_Utilities import *

def Page_Datahandler():
    st.title("Data Handler")
    st.write("Point data and graph are all required for training and predicting.")

def Page_Datahandler():
    st.title("Data Handler")
    st.write("Point data and graph are all required for training and predicting.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h1 style="font-size: 24px;">Point Data status</h1>', unsafe_allow_html=True)
        points_placeholder = st.empty()
    with col2:
        st.markdown('<h1 style="font-size: 24px;">Graph Data status</h1>', unsafe_allow_html=True)
        graph_placeholder = st.empty()

    while True:
        with points_placeholder.container():
            Update_Points()  # 在占位符中更新点数据
        with graph_placeholder.container():
            Update_Graph()  # 在占位符中更新图数据

        time.sleep(10)  # 每10秒更新一次，根据需求调整时间间隔

    SUB_PAGES = {
        "Point Data": Page_DataPoints,
        "Graph Data": Page_DataGraph
    }
    sub_selection = st.radio("选择子页面", list(SUB_PAGES.keys()))
    sub_page = SUB_PAGES[sub_selection]
    sub_page()