import h5py
from Data_Tool.Utilities.Data_meta import Data_meta
import numpy as np

def load_and_transpose(file_paths):
    transposed_data = []
    for file_path in file_paths:
        data = np.load(file_path)
        transposed_data.append(np.transpose(data, (3, 0, 1, 2)))
    return transposed_data

def data_creation(file_name,raw_data_transposed, transposed_data):
    with h5py.File(file_name, 'w') as h5f:
        h5f.create_dataset('raw', data=raw_data_transposed)
        h5f.create_dataset('label', data=transposed_data)

#===================================================================
raw_data_original = np.load('../../Data/raw/Data_42a0704d.npy')

file_paths = [
    '../../Data/raw/Data_fc96fe87.npy',
    '../../Data/raw/Data_b04c7fdf.npy',
    '../../Data/raw/Data_2eb898ba.npy',
    '../../Data/raw/Data_bdc8ddb1.npy'
]

hdf5_filename_1 = f'../../Data/Train_1/data_fc96fe87.h5'
hdf5_filename_2 = f'../../Data/Train_2/data_b04c7fdf.h5'
hdf5_filename_3 = f'../../Data/Train_3/data_2eb898ba.h5'
hdf5_filename_4 = f'../../Data/Train_4/data_bdc8ddb1.h5'

transposed_data_list = load_and_transpose(file_paths)

label_1_data_transposed = transposed_data_list[0]
label_2_data_transposed = transposed_data_list[1]
label_3_data_transposed = transposed_data_list[2]
label_4_data_transposed = transposed_data_list[3]

x, y, z, _ = raw_data_original.shape
raw_data = np.zeros((x,y,z,2), dtype=int)
for i in range(x):
    for j in range(y):
        for k in range(z):
            if raw_data_original[i, j, k, 0] == 1:
                raw_data[i, j, k] = [1, 0]
            else:
                raw_data[i, j, k] = [0, 1]
raw_data_transposed = np.transpose(raw_data, (3, 0, 1, 2))

data_creation(hdf5_filename_1, raw_data_transposed, label_1_data_transposed)
data_creation(hdf5_filename_2, raw_data_transposed, label_2_data_transposed)
data_creation(hdf5_filename_3, raw_data_transposed, label_3_data_transposed)
data_creation(hdf5_filename_4, raw_data_transposed, label_4_data_transposed)









        


    