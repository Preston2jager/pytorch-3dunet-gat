import torch

# 加载 checkpoint 文件
checkpoint_path = 'C:/Users/prest/Downloads/best_checkpoint.pytorch'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 查看 model_state_dict 中每个权重的形状
print("Model State Dict - Weights Shape:")
for layer_name, weight_tensor in checkpoint['model_state_dict'].items():
    print(f"Layer: {layer_name} | Shape: {weight_tensor.shape}")