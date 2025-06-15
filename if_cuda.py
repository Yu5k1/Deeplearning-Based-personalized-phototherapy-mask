import torch
print(torch.cuda.is_available())  # 很可能返回 False
print(torch.cuda.get_device_name(0))  # 如果 False，这里会报错