import tempfile,h5py,subprocess
import streamlit as st
from streamlit_modal import Modal
import numpy as np
from Utilities.Data_Utilities import *
from Utilities.Global import *

def Page_Datahandler():
    container1 = st.container()
    container2 = st.container()
    with container1:
        st.title("Step 1: Point Data ")
        uploaded_file = st.file_uploader("Start with upload the point data file in npy format")
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                Points_Dir = tmpfile.name
            if st.button("Analysis"):
                array = np.load(Points_Dir)
                x_dim, y_dim, z_dim = array.shape[:3]
                Total_points = x_dim * y_dim * z_dim
                classes = array.shape[-1] - 1
                invalid_count = np.sum(array[..., 0] == 1)
                valid_percentage = ((Total_points - invalid_count) / Total_points) * 100
                Evas = Data_Evaluation(valid_percentage)
                st.write('Total point data contained: ', Total_points)
                st.write('Total classification: ', classes)
                st.write('Valid Data: ', Total_points-invalid_count)
                st.write('Occupancy: {:.2f}%'.format(valid_percentage))
                st.write(Evas)
            if st.button('Render'):
                subprocess.run(['python', './Data_Render.py', '--file', Points_Dir], check=True)
            if st.button('Save as H5 file'):
                identifier = generate_hashed_timestamp()
                if 'identifier' not in st.session_state:
                    st.session_state.identifier = identifier
                hdf5_filename = f'../Data/Train/data_{identifier}.h5'
                with h5py.File(hdf5_filename, 'w') as h5f:
                    raw_data = np.load(Points_Dir)
                    x, y, z, _ = raw_data.shape
                    label_data = np.zeros((x,y,z,2), dtype=int)
                    for i in range(x):
                        for j in range(y):
                            for k in range(z):
                                if raw_data[i, j, k, 0] == 1:
                                    label_data[i, j, k] = [1, 0]
                                else:
                                    label_data[i, j, k] = [0, 1]
                    raw_data_transposed = np.transpose(raw_data, (3, 0, 1, 2))
                    label_data_transposed = np.transpose(label_data, (3, 0, 1, 2))
                    h5f.create_dataset('raw', data=raw_data_transposed)
                    h5f.create_dataset('label', data=label_data_transposed)
                    st.success('Point data file saved successfully! ✅')
    with container2:
        st.title("Step 2: Graph Data ")
        st.write("""
        Input your graph data below. Each line represents a node or an edge.
        - For nodes with parameters, input the node name followed by key-value pairs (e.g., `A color red size 10`).
        - For edges, input the node names separated by a space (e.g., `A B`).\\
        A Name 1 Size 22\\
        B Name 2 Size 10\\
        C Name 3 Size 5\\
        D Name 4 Size 8.\\
        E Name 5 Size 8\\
        A B\\
        A C\\
        A D\\
        A E\\
        E D
        """)
        input_text = st.text_area("Input Graph Data", height=200)

        if st.button("Create Graph"):
            if input_text:
                G = create_graph_from_text(input_text)
                node_colors = {node: get_random_color() for node in G.nodes()}
                node_size = 300
                fig, ax = plt.subplots()
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=False, node_color=[node_colors[node] for node in G.nodes()], node_size=node_size, edge_color='gray', ax=ax)
                labels = {node: f"{node}\n" + "\n".join([f"{k}: {v}" for k, v in data.items()]) for node, data in G.nodes(data=True)}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
                st.pyplot(fig)
            else:
                st.error("Please input graph data.")
        
        if st.button("Save Graph"):
            if input_text:
                G = create_graph_from_text(input_text)
                nodes, edges = graph_to_tensors(G)
                if 'identifier' in st.session_state:
                    identifier = st.session_state.identifier
                    nodes_filename = f'../Data/Train/nodes_{identifier}.pt'
                    edges_filename = f'../Data/Train/edges_{identifier}.pt'
                    torch.save(nodes, nodes_filename)
                    torch.save(edges, edges_filename)
                    st.success("graph data saved")
                else:
                    st.write("Identifier not found.")



       

        


        


    