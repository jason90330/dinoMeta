'''
For CelebA-Spoof
'''
from torch.utils.data import Dataset
from imutils import paths
from PIL import Image 
import numpy as np
import torch
import random
import json
import os

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def load_mat(path):
    mat = np.loadtxt(path)
    return torch.tensor(mat,dtype=torch.float32)

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for catimages, labels in data_loader:
            yield (catimages, labels)

class customData(Dataset):
    def __init__(self, illumination_domain, img_path, txt_path, json_path, phase = '',data_transforms=None, loader = default_loader):
        with open(json_path, "r") as input_json:
            jf = json.loads(input_json.read())

        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.Random(4).shuffle(lines)
            # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
            self.img_name = []
            self.img_label = []
            lengthOfTrain = int(len(lines)*0.8)
            if phase=='train':
                for line in lines[:lengthOfTrain]:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)
            elif phase=='val':
                for line in lines[lengthOfTrain:]:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)
            elif phase=="ssl":
                for line in lines:
                    pth = line.strip().split(' ')[0]
                    path = os.path.join(img_path, (pth[11:]))
                    # spoofType = jf[pth][40]
                    label = int(line.strip().split(' ')[-1])
                    
                    illumination = jf[pth][41]
                    if illumination == illumination_domain:
                        if os.path.isfile(path):
                            self.img_name.append(path)
                            self.img_label.append(label)
                        else:
                            print(path)
                    else:
                        continue


            # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.phase = phase
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            label = self.img_label[item]
            img = self.loader(img_name)

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms(img)
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, label
        except:
            print(item)