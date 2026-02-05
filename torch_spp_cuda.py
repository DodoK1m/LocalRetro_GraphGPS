import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count())
