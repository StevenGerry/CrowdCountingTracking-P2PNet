
# Project Setup
## Create a blank conda virtual environment
For example, python 3.10
## Download torch and cuda toolkit
source - https://pytorch.org/get-started/locally/
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
## Download all other dependencies
Since torch is already downloaded, it needs to be removed from requirements.txt
or recursively download libraries from the prepared requirements_dev.txt
## Check
```python
import torch

torch.cuda.is_available()
```
Should be True

# Training Locally
1. Download the dataset from kaggle (https://www.kaggle.com/datasets/tthien/shanghaitech) and place it in the root directory
2. Download weights for vgg and place them in the models_storage folder (only vgg16_bn is needed)
```python
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
```
3. Run the parse_dataset.py script (once for train, once for test) to prepare the dataset
```python
data_source_root = r'D:\Projects\CrowdCountingTracking-P2PNet\ShanghaiTech\part_B\train_data'
output_dir = r'D:\Projects\CrowdCountingTracking-P2PNet\DATA_ROOT'
dataset_part = 'train'
```
and
```python
data_source_root = r'D:\Projects\CrowdCountingTracking-P2PNet\ShanghaiTech\part_B\test_data'
output_dir = r'D:\Projects\CrowdCountingTracking-P2PNet\DATA_ROOT'
dataset_part = 'test'
```