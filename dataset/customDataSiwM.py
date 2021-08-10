'''
For Siw-M
'''
from torch.utils.data import Dataset
from imutils import paths
from PIL import Image 
import numpy as np
import torch
import random
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

class customData(Dataset):
    def __init__(self, img_path, txt_path, phase = '',data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            foldLists = input_file.readlines()
            img_paths = list(paths.list_images(img_path))
            random.Random(4).shuffle(img_paths)
            # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
            self.img_name = []
            self.img_label = []
            lengthOfTrain = int(len(img_paths)*0.8)
            spoof_types =   { 
                        'Replay': 1,
                        'Paper': 2,
                        'HalfMask': 3,
                        'SiliconeMask': 4,
                        'TransparentMask': 5,
                        'PaperMask': 6,
                        'MannequinHead': 7,
                        'Obfuscation': 8,
                        'Impersonation': 9,
                        'Cosmetic': 10,
                        'FunnyeyeGlasses': 11,
                        'PaperGlasses': 12,
                        'PaperCut': 13,
                        }
            if phase=='train':
                for path in img_paths:#不能先吃
                    for line in foldLists:
                        trainFolder = line.strip("\n")
                        if "Live" in path and "Train" in path:
                            self.img_name.append(path)
                            self.img_label.append(0)
                            break
                        elif trainFolder in path:
                            name = path.split('/')[-3]#只保留類別名稱
                            self.img_name.append(path)
                            label = spoof_types[name]
                            self.img_label.append(label)
                            break
                        else:
                            # print(path)
                            continue                        
            elif phase=='val':
                for path in img_paths[lengthOfTrain:]:
                    for line in foldLists:
                        validFolder = line.strip("\n")
                        if "Live" in path and "Train" in path:
                            self.img_name.append(path)
                            self.img_label.append(0)
                            break
                        elif validFolder in path:
                            name = path.split('/')[-3]#只保留類別名稱
                            self.img_name.append(path)
                            label = spoof_types[name]
                            self.img_label.append(label)
                            break
                        else:
                            # print(path)
                            continue
            elif phase=="test":
                for path in img_paths:
                    try:
                        for idx, line in enumerate(foldLists):
                            trainFolder = line.strip("\n")
                            if "Live" in path and "Train" not in path:
                                self.img_name.append(path)
                                self.img_label.append(0)
                                break                            
                            elif trainFolder in path or "Train" in path:
                                break # 任何一個 train folder 有出現在 path 當中，代表不是 test data
                            elif idx == len(foldLists)-1: # 所有 train folder 都沒出現在 path 當中，代表是 test data
                                name = path.split('/')[-3]#只保留類別名稱
                                self.img_name.append(path)
                                label = spoof_types[name]
                                self.img_label.append(label)
                                break
                            else:
                                continue
                    except:
                        print(path)
            elif phase=="test_lack_8":
                for path in img_paths:
                    for line in foldLists:
                        testFolder = line.strip("\n")
                        if "Live" in path and "Test" in path:
                            self.img_name.append(path)
                            self.img_label.append(0)
                            break
                        elif testFolder in path:
                            name = path.split('/')[-3]#只保留類別名稱
                            self.img_name.append(path)
                            label = spoof_types[name]
                            self.img_label.append(label)
                            break
                        else:
                            # print(path)
                            continue
            elif phase=="ssl":
                for path in img_paths:#不能先吃
                    for line in foldLists:
                        trainFolder = line.strip("\n")
                        if "Live" in path and "Train" in path:
                            self.img_name.append(path)
                            self.img_label.append(0)
                            break
                        elif trainFolder in path:
                            self.img_name.append(path)
                            self.img_label.append(1)
                            break
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


class customData_align_mat(Dataset):
    def __init__(self, img_path, mat_path, txt_path, dataset = '',data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.Random(4).shuffle(lines)
            # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
            self.img_name = []
            self.mat_path = []
            self.img_label = []
            lengthOfTrain = int(len(lines)*0.8)
            if dataset=='train':
                for line in lines[:lengthOfTrain]:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    mpath = path.replace("trainSquareCropped", "trainCroppedSquareTxt")
                    mpath = mpath.replace("jpg", "txt")
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.mat_path.append(mpath)
                        self.img_label.append(label)
                    else:
                        print(path)
                                        
            elif dataset=='val':
                for line in lines[lengthOfTrain:]:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    mpath = path.replace("trainSquareCropped", "trainCroppedSquareTxt")
                    mpath = mpath.replace("jpg", "txt")
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.mat_path.append(mpath)
                        self.img_label.append(label)
                    else:
                        print(path)
            # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader
        self.load_mat = load_mat

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            img = self.loader(img_name)
            path_name = self.mat_path[item]
            mat = self.load_mat(path_name)
            label = self.img_label[item]

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms[self.dataset](img)
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, mat, label
        except:
            print(item)