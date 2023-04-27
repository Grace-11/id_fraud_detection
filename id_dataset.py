import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
from math import ceil
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from augment_collection import ColorJitter, Lighting

#NUMBER_OF_CLASSES = 2 # 2: real, fake / 3: real, screen, print

ID_STATE = ["real", "screen", "print"]

class torchvision_aug():
    def __init__(self, img_H, img_W):
        self.img_H = img_H
        self.img_W = img_W
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    @staticmethod
    def cv2_jpg(img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]
    @staticmethod
    def pil_jpg(img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img
    def rotate_align_pad_to_minimum_size(self, image):
        w, h = image.size
        if w < h:
            image = image.rotate(90, expand=1)
        w, h = image.size       
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])
    def get_aug20(self):  # CUTMIX github에 있는 imagenet augmentation
            jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
            lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
            transform = T.Compose([
                T.Lambda(self.rotate_align_pad_to_minimum_size),
                T.Resize((self.img_H, self.img_W)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                jittering,
                lighting,
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return transform

class torchvision_test_aug():
    def __init__(self, img_H, img_W):
        self.img_H = img_H
        self.img_W = img_W
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    @staticmethod
    def cv2_jpg(img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]
    @staticmethod
    def pil_jpg(img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img
    def rotate_align_pad_to_minimum_size(self, image):
        w, h = image.size
        if w < h:
            image = image.rotate(90, expand=1)
        w, h = image.size       
        h_diff = h - self.img_H
        w_diff = w - self.img_W
        h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
        w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
        if h_pad == 0 and w_pad == 0:
            return image
        else:
            return T.functional.pad(image, [h_pad, w_pad])
    def get_aug5(self):
            transform = T.Compose([
                T.Lambda(self.rotate_align_pad_to_minimum_size),
                T.Resize((self.img_H, self.img_W)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return transform

class IdDataset(Dataset):
    def __init__(self, cfg, type, transform):
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        with open(self.data_cfg["data_json"]) as f:
            files = json.load(f)

        keys = list(files.keys())
        self.ids = []
        self.labels = []
        #self.list_of_files = list(files.values())

        #assert(type == "train" or "test")

        for key in keys:
            if type in key:   # only train or only test
                labels = files[key] #self.labels
                val = 0 if 'real' in key else 1
                for label in labels:
                    self.ids.append(label)
                    self.labels.append(val)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.ids[index]).convert("RGB")
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def get_labels(self):
        return self.labels
    
'''
    img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        class_ = self.classes[idx]
        img = self.transform(img)
        return img, label, class_

'''