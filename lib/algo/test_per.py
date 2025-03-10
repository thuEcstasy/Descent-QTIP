import torch

tensor = torch.tensor([[1, 2, 3], 
                       [4, 5, 6], 
                       [7, 8, 9]])

# Step 1: 将二维张量展平成一维
flat_tensor = tensor.flatten()

permutation = torch.randperm(flat_tensor.size(0))
shuffled_tensor = flat_tensor[permutation]

# Step 3: 记录恢复顺序的索引
inverse_permutation = torch.argsort(permutation)

# 打印结果
print("Flattened tensor:", flat_tensor)
print("Shuffled tensor:", shuffled_tensor)
print("Recovered tensor:", shuffled_tensor[inverse_permutation].view_as(tensor))
