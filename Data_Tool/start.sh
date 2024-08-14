#!/bin/bash

# 切换到项目目录
cd .. || { echo "Failed to change directory"; exit 1; }

# 卸载旧版的 pytorch3dunet
pip uninstall -y pytorch3dunet || { echo "Failed to uninstall pytorch3dunet"; exit 1; }

# 安装 pytorch3dunet
python ./setup.py install || { echo "Failed to install pytorch3dunet"; exit 1; }

# 切换到 Data_Tool 目录
cd Data_Tool || { echo "Failed to change to Data_Tool directory"; exit 1; }

# 运行 train3dunet 命令
train3dunet --config ../resources/3DUnet_multiclass/train_config.yaml || { echo "Failed to run train3dunet"; exit 1; }

echo "Script executed successfully"
