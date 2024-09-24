import numpy as np
import h5py 

#=================================
#Loading data file
file_path_1 = 'C:/Users/prest/Downloads/predictions/data_bdc8ddb1_predictions_1.h5'
file_path_2 = 'C:/Users/prest/Downloads/predictions/data_bdc8ddb1_predictions_3.h5'

with h5py.File(file_path_1, 'r') as h5_file_1:
    array_1 = h5_file_1['predictions'][:]

with h5py.File(file_path_2, 'r') as h5_file_2:
    array_2 = h5_file_2['predictions'][:]

# 比较两个数组对应元素是否相同
comparison = array_1 == array_2

# 计算相同元素的百分比
percentage_identical = np.mean(comparison) * 100

print(f"两个数组中相同元素的百分比: {percentage_identical:.2f}%")