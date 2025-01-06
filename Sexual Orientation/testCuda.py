import os
# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(7)))

# Now, when you import and use any libraries that use CUDA (e.g., PyTorch, TensorFlow),
#they will only see and use GPUs 0 and 1.
import torch
print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")