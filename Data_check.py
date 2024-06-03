import h5py

# 打开HDF5文件
file_path = './Data/data_20240603_143707.h5'  # 替换为您的文件路径
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的每个数据集
    for dataset_name in file:
        dataset = file[dataset_name]
        # 打印数据集的名字和尺寸
        print(f"Dataset Name: {dataset_name}")
        print(f"Shape: {dataset.shape}")
        print(f"Size: {dataset.size}")
        print(f"Dtype: {dataset.dtype}")
