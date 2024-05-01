import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
from helpers import *
from torchvision.transforms import v2, PILToTensor
from torch.utils.data import random_split

class PoolDatasetV2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.labeled_images = get_images_labeled(self.root_dir.replace('images','labels'))
        self.retrieve_data()

    def retrieve_data(self):
        self.images = []
        self.labels = []
        
        for file in os.listdir(self.root_dir):
            
            full_image_path = self.root_dir + "\\" + file
            label_name = change_image_to_label(file)


            # Deal with the image
            img = Image.open(full_image_path).convert('RGB')

            # Transform to tensor
            if "PIL" in str(type(img)):
                img = PILToTensor()(img)
            else:
                img = torch.Tensor(img)

            self.images.append(img)


            # Deal with the label
            if label_name in self.labeled_images:
                boxes = getBoxes(self.root_dir.replace('images','labels') + "\\" + label_name)
                self.labels.append([torch.Tensor(box) for box in boxes])

            else:
                self.labels.append(torch.Tensor([]))

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        img = self.images[idx]
        boxes = self.labels[idx]

        old_width = img.shape[1]

        if self.transform:
            img = torch.Tensor(self.transform(img))
            
            factor = old_width / img.shape[1]

            if len(boxes) > 0:
                boxes = list(map(lambda l: torch.tensor([v / factor for v in l]), boxes))
                #max_boxes = max(len(boxes), 1)  # Ensure at least 1 box
                #pad_boxes = F.pad(torch.stack(boxes), (0, 0, 0, max_boxes - len(boxes)), value=-1)
                #boxes = pad_boxes.unbind(0)
        else:
            boxes = torch.Tensor(boxes)

        return img, boxes
    
    def split_Data(self, n_test=0.33):
        test_size = round(n_test * len(self.images))
        train_size = len(self.images) - test_size

        return random_split(self, [train_size, test_size])


# EXAMPLE
# transforms = v2.Compose([
#     v2.Resize((224, 224)),
#     v2.ToImage(),
#     #v2.RandomHorizontalFlip(p=1),
#     v2.ToDtype(torch.float32, scale=True),
#     #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# ds = PoolDatasetV2(".\\dataset\\images", transform=transforms)
# # ds = PoolDatasetV2(".\\dataset\\images", transform=None)
# item = ds[7]
# print(item[1])