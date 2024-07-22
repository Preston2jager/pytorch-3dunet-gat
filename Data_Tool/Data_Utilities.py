import streamlit as st
from streamlit_modal import Modal

def Display_graph(graph_data,edge_index):
    modal = Modal(
        "Demo Modal", 
        key="demo-modal",
        padding=20,    # default value
        max_width=744  # default value
    )
        modal.open()
    if modal.is_open():
        with modal.container():
            st.write(graph_data)
            st.write(edge_index)
            value = st.checkbox("Check me")
            st.write(f"Checkbox checked: {value}")