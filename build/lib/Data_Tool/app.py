import tempfile,h5py,subprocess,shutil,torch,os
import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Utilities.Data_meta import Data_meta
from Utilities.Data_Utilities import create_graph_from_text, get_random_color, graph_to_tensors, Data_Evaluation, clear_folders, generate_hashed_timestamp
    

def single_page_app():
    st.title("Data Handler and Training Page")

    # Step 1: Point Data
    st.header("Step 1: Point Data")
    uploaded_file = st.file_uploader("Upload the point data file in npy format")
    Data_meta_instance = Data_meta.get_instance()
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            Data_meta_instance.set_Points_Dir(tmpfile.name)

        if st.button("Analyze"):
            array = np.load(Data_meta_instance.Points_Dir)
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
            subprocess.run(['python', './Data_Render.py', '--file', Data_meta_instance.Points_Dir], check=True)

        if st.button('Save as H5 file'):
            clear_folders()
            identifier = generate_hashed_timestamp()
            if 'identifier' not in st.session_state:
                st.session_state.identifier = identifier
            hdf5_filename = f'../Data/Train/data_{identifier}.h5'
            Data_meta_instance.set_hashcode(identifier)
            st.success(identifier)
            st.success(Data_meta_instance.hashcode)
            
            with h5py.File(hdf5_filename, 'w') as h5f:
                label_data = np.load(Data_meta_instance.Points_Dir)
                x, y, z, _ = label_data.shape
                raw_data = np.zeros((x, y, z, 2), dtype=int)
                for i in range(x):
                    for j in range(y):
                        for k in range(z):
                            if label_data[i, j, k, 0] == 1:
                                raw_data[i, j, k] = [1, 0]
                            else:
                                raw_data[i, j, k] = [0, 1]
                
                raw_data_transposed = np.transpose(raw_data, (3, 0, 1, 2))
                label_data_transposed = np.transpose(label_data, (3, 0, 1, 2))
                h5f.create_dataset('raw', data=raw_data_transposed)
                h5f.create_dataset('label', data=label_data_transposed)
                st.success('Point data file saved successfully! ✅')
                
            new_directory = '../Data/Val/'
            new_filename = f'Val_{identifier}.h5'
            new_file_path = os.path.join(new_directory, new_filename)
            try:
                shutil.copy(hdf5_filename, new_file_path)
                st.success(f'File copied and renamed successfully to {new_file_path} ✅')
            except Exception as e:
                st.error(f'Failed to copy the file: {e}')

    # Step 2: Graph Data
    st.header("Step 2: Graph Data")
    st.write("""
    Input your graph data below. Each line represents a node or an edge.
    - For nodes with parameters, input the node name followed by key-value pairs (e.g., `A color red size 10`).
    - For edges, input the node names separated by a space (e.g., `A B`).
    A Name Living_room Size 22
    B Name Bedroom Size 10
    C Name Bathroom Size 5
    D Name Kitchen Size 8.
    E Name Dining_room Size 8
    A B
    A C
    A D
    A E
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
        st.write(G.nodes)
        st.write(G.edges)

    if st.button("Save Graph"):
        if input_text:
            G = create_graph_from_text(input_text)
            nodes, edges = graph_to_tensors(G)
            if 'identifier' in st.session_state:
                identifier = st.session_state.identifier
                nodes_filename = f'../Data/Train/nodes.pt'
                edges_filename = f'../Data/Train/edges.pt'
                torch.save(nodes, nodes_filename)
                torch.save(edges, edges_filename)
                st.success("Graph data saved")
            else:
                st.write("Identifier not found.")

if __name__ == "__main__":
    single_page_app()
