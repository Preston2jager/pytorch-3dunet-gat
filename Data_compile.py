import h5py
import numpy as np
from datetime import datetime

# 获取当前日期和时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 定义文件名，包括路径和时间戳
filename = f'./Data/data_{current_time}.h5'

# 创建 HDF5 文件
with h5py.File(filename, 'w') as h5f:
    # 假设你已经有了要保存的数据
    raw_data = np.load('./Data/raw.npy')
    label_data = np.load('./Data/label.npy')

    # 创建数据组
    h5f.create_dataset('raw', data=raw_data)
    h5f.create_dataset('label', data=label_data)

print(f"数据已经被保存为 HDF5 格式，文件名为：{filename}")
