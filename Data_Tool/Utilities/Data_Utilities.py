import os,hashlib,random,torch
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Utilities.Global import *
from datetime import datetime

def Check_Points(Points_Dir):
    if Points_Dir != "":
        Points_status = True
    else:
        Points_status = False
    return Points_status

def Check_Graph():
    Graph_status = True
    return Graph_status

def Update_Points():
    global Points_Dir
    r = Check_Points(Points_Dir)
    if r:
        st.markdown(
            '<p style="color:green;">Point Data Ready</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:red;">Points Data Not Ready</p>',
            unsafe_allow_html=True
        )
    
def Update_Graph():
    r = Check_Graph()
    if r:
        st.markdown(
            '<p style="color:green;">Graph Data Ready</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:red;">Graph Data Not Ready</p>',
            unsafe_allow_html=True
        )

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('选择一个文件', filenames)
    return os.path.join(folder_path, selected_filename)

def Data_Evaluation(valid_percentage):
    if valid_percentage > 50:
        return "Great data"
    elif valid_percentage > 30:
        return "OK data"
    elif valid_percentage >10:
        return "Not Ideal"
    else:
        return "Bad data"
    
def generate_hashed_timestamp():
    timestamp = datetime.now().isoformat()
    hash_object = hashlib.sha256(timestamp.encode())
    short_identifier = hash_object.hexdigest()[:8]
    return short_identifier

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

def graph_to_tensors(G):
    # 获取所有节点的特征字典并将所有数值转换为浮点类型
    node_features_dict = {node: {k: float(v) for k, v in G.nodes[node].items()} for node in G.nodes()}
    # 确定所有可能的特征
    all_keys = set()
    for features in node_features_dict.values():
        all_keys.update(features.keys())
    # 确保 'Node name' 是第一个特征
    all_keys = ['Node name'] + [key for key in all_keys if key != 'Node name']
    # 构建节点特征矩阵
    node_features = []
    for node in G.nodes():
        feature_vector = []
        for key in all_keys:
            if key == 'Node name':
                feature_vector.append(float(hash(node)))  # 将节点名称哈希为一个数值并转换为浮点类型
            else:
                feature_vector.append(node_features_dict[node].get(key, 0.0))
        node_features.append(feature_vector)
    # 映射节点到索引
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    # 构建边索引矩阵
    edge_index = []
    for edge in G.edges():
        edge_index.append([node_mapping[edge[0]], node_mapping[edge[1]]])
    edge_index = np.array(edge_index).T
    # 转换为PyTorch张量
    nodes = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # 移除第一列
    nodes_final = nodes[:, 1:]

    return nodes_final, edge_index