import click, h5py, torch, glob
import tkinter as tk
import numpy as np
from tkinter import filedialog
from datetime import datetime

def select_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename()  
    return file_path

@click.group(invoke_without_command=True)
@click.pass_context

@click.group()
def cli():
    pass

def get_user_choice():
    choices = ['1', '2', '3', '4','5','6','7']  # 定义可用的选择
    prompt_text = (
        "3D-U-Net-GAT data handler\n"
        "Chooes your task:\n"
        "1: Display current graph\n"
        "2: Add a node\n"
        "3: Add an edge\n"
        "4: Reset graph\n"
        "5: Export data set\n"
        "6: Check HDF5 file status\n"
        "7: Exit\n"
    )
    choice = click.prompt(prompt_text, type=click.Choice(choices))
    return choice

def Display_graph(graph_data,edge_index):
    """Display graph information"""
    print("Nodes:")
    print(graph_data)
    print("Edges:")
    print(edge_index)

def Reset_data(graph_data,edge_index):
    """Reset graph data set"""
    graph_data = torch.empty(0, 4)
    edge_index = torch.empty(0, 2,dtype=torch.long)
    return graph_data,edge_index

def add_node():
    """Add a node"""
    node_name = click.prompt('Room Name', type=str)
    node_type = click.prompt('Room Size', type=str)
    # 使用输入的信息
    click.echo(f"正在添加节点: {node_name}, 类型: {node_type}")

def add_edge():
    """Add a edge"""
    node_name = click.prompt('Room Name', type=str)
    node_type = click.prompt('Room Size', type=str)
    # 使用输入的信息
    click.echo(f"正在添加节点: {node_name}, 类型: {node_type}")

def Check_hdf5():
    """Check the current hdf5 status"""
    Train_file_paths = glob.glob(r"./Data/Train/data_*.h5")
    Val_file_paths = glob.glob(r"./Data/Val/data_*.h5")
    print(f"----------Training files----------")
    for file_path in Train_file_paths:
        click.echo(f"Selected file: {file_path}")
        print(f"----------Content----------")
        with h5py.File(file_path, 'r') as file:
            for dataset_name in file:
                dataset = file[dataset_name]
                print(f"Dataset Name: {dataset_name}")
                print(f"Shape: {dataset.shape}")
                print(f"Size: {dataset.size}")
                print(f"Dtype: {dataset.dtype}")
                print(f"----------End----------")
    print(f"----------Validation files----------")
    for file_path in Val_file_paths:
        click.echo(f"Selected file: {file_path}")
        print(f"----------Content----------")
        with h5py.File(file_path, 'r') as file:
            for dataset_name in file:
                dataset = file[dataset_name]
                print(f"Dataset Name: {dataset_name}")
                print(f"Shape: {dataset.shape}")
                print(f"Size: {dataset.size}")
                print(f"Dtype: {dataset.dtype}")
                print(f"----------End----------")

def Export_data_set(graph_data,edge_index):
    """Compile the Numpy file to hdf5 file"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_filename = f'./Data/Graph/graph_{current_time}.h5'
    hdf5_filename = f'./Data/data_{current_time}.h5'
    graph_data_np = graph_data.numpy()
    edge_index_np = edge_index.numpy()
    with h5py.File(hdf5_filename, 'w') as h5f:
        raw_data = np.load('./Data/raw.npy')
        raw_data_expanded = np.zeros((400, 400, 120, 6), dtype=int)
        mapping = np.array([
            [1, 0, 0, 0, 0, 0],  # This maps to [1, 0]
            [0, 0, 0, 0, 0, 1]   # This maps to [0, 1]
        ])
        raw_data_expanded = mapping[raw_data[..., 1]]
        raw_data_transposed = np.transpose(raw_data_expanded, (3, 0, 1, 2))
        label_data = np.load('./Data/label.npy')
        label_data_transposed = np.transpose(label_data, (3, 0, 1, 2))
        h5f.create_dataset('raw', data=raw_data_transposed)
        h5f.create_dataset('label', data=label_data_transposed)
    with h5py.File(graph_filename, 'w') as f:
        f.create_dataset('graph_data', data=graph_data_np)
        f.create_dataset('edge_index', data=edge_index_np)
    print(f"----------Start----------")
    print(f"Graph has been created as {graph_filename}")
    print(f"Data has been created with HDF5 format as {hdf5_filename}")
    print(f"----------End----------")

def exit_cli():
    """Simply exit the tool"""
    click.echo("Exiting the program...")
    raise click.Abort()

if __name__ == '__main__':
    graph_data = torch.empty(0, 4)
    edge_index = torch.empty(0, 2,dtype=torch.long)
    while True:
        user_choice = get_user_choice()
        if user_choice == '1':
            Display_graph(graph_data,edge_index)
        elif user_choice == '2':
            add_node()
        elif user_choice == '3':
            add_edge()
        elif user_choice == '4':
            Reset_data(graph_data,edge_index)
        elif user_choice == '5':
            Export_data_set(graph_data,edge_index)
        elif user_choice == '6':
            Check_hdf5()
        elif user_choice == '7':
            exit_cli()

            