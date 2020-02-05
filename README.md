# Tensorflow_TPU_modules
One stop destination for everything related to using Tensorflow on TPUs (repo still updating)

Models: 

Contains model definitions of different architectures (implemented in Tensorflow)
1. Inception3D
2. UNET
3. ResUNET
4. ResNext

Augmentations: Contains TF augmentations for 2D image (H, W, C) and 3D volume (D, H, W, C)

Utils: Layer definitions tailored to TPU hardware. Folder contains script that ensures that BatchNorm statistics are aggregated from all cores of TPU and not just the first core (default TPU setting)
