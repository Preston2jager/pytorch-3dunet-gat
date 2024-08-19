import h5py
from Data_Tool.Utilities.Data_meta import Data_meta
import numpy as np

def load_and_transpose(file_paths):
    transposed_data = []
    for file_path in file_paths:
        data = np.load(file_path)
        transposed_data.append(np.transpose(data, (3, 0, 1, 2)))
    return transposed_data

file_paths = [
    '../Data/raw/Data_fc96fe87.npy',
    '../Data/raw/Data_b04c7fdf.npy',
    '../Data/raw/Data_2eb898ba.npy',
    '../Data/raw/Data_bdc8ddb1.npy'
]

transposed_data_list = load_and_transpose(file_paths)

hdf5_filename_1 = f'../Data/Train_1/data_fc96fe87.h5'
hdf5_filename_2 = f'../Data/Train_2/data_b04c7fdf.h5'
hdf5_filename_3 = f'../Data/Train_3/data_2eb898ba.h5'
hdf5_filename_val = f'../Data/Train_val/data_bdc8ddb1.h5'
hdf5_filename_pre = f'../Data/Train_pre/data_bdc8ddb1.h5'

label_1_data_transposed = transposed_data_list[0]
label_2_data_transposed = transposed_data_list[1]
label_3_data_transposed = transposed_data_list[2]
val_data_transposed = transposed_data_list[3]

label_data = np.load('../Data/raw/Data_42a0704d.npy')
x, y, z, _ = label_data.shape
raw_data = np.zeros((x,y,z,2), dtype=int)
for i in range(x):
    for j in range(y):
        for k in range(z):
            if label_data[i, j, k, 0] == 1:
                raw_data[i, j, k] = [1, 0]
            else:
                raw_data[i, j, k] = [0, 1]
raw_data_transposed = np.transpose(raw_data, (3, 0, 1, 2))


def data_creation(file_name, transposed_data,raw_data_transposed):
    with h5py.File(file_name, 'w') as h5f:
        h5f.create_dataset('raw', data=raw_data_transposed)
        h5f.create_dataset('label', data=transposed_data)

#data_creation(hdf5_filename_1, label_1_data_transposed, raw_data_transposed)
#data_creation(hdf5_filename_2, label_2_data_transposed, raw_data_transposed)
#data_creation(hdf5_filename_3, label_3_data_transposed, raw_data_transposed)
#data_creation(hdf5_filename_val, val_data_transposed, raw_data_transposed)
data_creation(hdf5_filename_pre, val_data_transposed)








        


    