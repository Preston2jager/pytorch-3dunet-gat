import tempfile,h5py,subprocess
import streamlit as st
import numpy as np
from Utilities.Data_Utilities import *
from Utilities.Global import *

def Page_DataPoints():
    st.title("Point Data ")
    uploaded_file = st.file_uploader("Upload the point data file in npy format")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            Points_Dir = tmpfile.name
        st.write('File saved at:', Points_Dir)

        if st.button("Analysis"):
            array = np.load(Points_Dir)
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
            subprocess.run(['python', './Data_Render.py', '--file', Points_Dir], check=True)

        if st.button('Save as H5 file'):
            identifier = generate_hashed_timestamp()
            hdf5_filename = f'../Data/Train/data_{identifier}.h5'
            with h5py.File(hdf5_filename, 'w') as h5f:
                raw_data = np.load(Points_Dir)
                x, y, z, _ = raw_data.shape
                label_data = np.zeros((x,y,z,2), dtype=int)
                for i in range(x):
                    for j in range(y):
                        for k in range(z):
                            if raw_data[i, j, k, 0] == 1:
                                label_data[i, j, k] = [1, 0]
                            else:
                                label_data[i, j, k] = [0, 1]
                raw_data_transposed = np.transpose(raw_data, (3, 0, 1, 2))
                label_data_transposed = np.transpose(label_data, (3, 0, 1, 2))
                h5f.create_dataset('raw', data=raw_data_transposed)
                h5f.create_dataset('label', data=label_data_transposed)
            st.success('File saved successfully! ✅') 
    