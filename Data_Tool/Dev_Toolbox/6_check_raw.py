import h5py

# 打开 HDF5 文件
file_path = '../Data/Train_pre/data_bdc8ddb1.h5'
with h5py.File(file_path, 'r') as h5_file:
    # 列出文件中的所有组
    print("Keys: %s" % h5_file.keys())
    
    # 获取特定的数据集
    dataset_name = 'raw'  # 替换为你实际的数据集名称
    data = h5_file[dataset_name][:]
    
    # 打印数据
    print(data)
    print(data.shape)
