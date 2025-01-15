import torch

print('GPU is available:', torch.cuda.is_available())
print('GPU Number:', torch.cuda.device_count())
try:
    print('GPU Name:', torch.cuda.get_device_name(0))
except:
    print('Cuda not/wrong installed!')
