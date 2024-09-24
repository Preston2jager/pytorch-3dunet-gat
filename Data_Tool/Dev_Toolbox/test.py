import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random

def create_graph_from_text(input_text):
    G = nx.Graph()
    for line in input_text.strip().split('\n'):
        if line.strip():
            parts = line.split()
            node = parts[0]
            if len(parts) > 2:
                node_params = {}
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        node_params[parts[i]] = parts[i + 1]
                G.add_node(node, **node_params)
            elif len(parts) == 1:
                G.add_node(node)
            elif len(parts) == 2:
                G.add_edge(parts[0], parts[1])
    return G

def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

st.title("Graph Data Input and Visualization")

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
        
        # Generate random colors for each node
        node_colors = {node: get_random_color() for node in G.nodes()}
        
        # Set a uniform size for all nodes
        node_size = 700

        fig, ax = plt.subplots()
        
        pos = nx.spring_layout(G)
        
        nx.draw(G, pos, with_labels=False, node_color=[node_colors[node] for node in G.nodes()], node_size=node_size, edge_color='gray', ax=ax)

        # Add node labels with their parameters
        labels = {node: f"{node}\n" + "\n".join([f"{k}: {v}" for k, v in data.items()]) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        st.pyplot(fig)
    else:
        st.error("Please input graph data.")
