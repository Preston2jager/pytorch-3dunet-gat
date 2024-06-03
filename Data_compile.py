import h5py
import numpy as np
from datetime import datetime

# Accquire data and time to create unique file name
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'./Data/data_{current_time}.h5'

with h5py.File(filename, 'w') as h5f:
    #Get numpy
    raw_data = np.load('./Data/raw.npy')
    #Mapping the last dimension to meet 3D Unet input requirement
    raw_data_expanded = np.zeros((400, 400, 120, 6), dtype=int)
    mapping = np.array([
        [1, 0, 0, 0, 0, 0],  # This maps to [1, 0]
        [0, 0, 0, 0, 0, 1]   # This maps to [0, 1]
    ])
    raw_data_expanded = mapping[raw_data[..., 1]]
    # transpose to meet 3D Unet input requirement
    raw_data_transposed = np.transpose(raw_data_expanded, (3, 0, 1, 2))
    label_data = np.load('./Data/label.npy')
    label_data_transposed = np.transpose(label_data, (3, 0, 1, 2))
    
    h5f.create_dataset('raw', data=raw_data_transposed)
    h5f.create_dataset('label', data=label_data_transposed)

print(f"Data has been created with HDF5 format as {filename}")
