import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class PoolDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.boxes = []
        
        self.parse_xml()
    
    def parse_xml(self):
        xml_files = [file for file in os.listdir(os.path.join(self.root_dir, 'labels')) if file.endswith('.xml')]

        for xml_file in xml_files:
            tree = ET.parse(os.path.join(self.root_dir, 'labels',xml_file))
            root = tree.getroot()

            image_path = None
            boxes = []

            for elem in root:
                
                if elem.tag == 'path':
                    image_path = elem.text

                if elem.tag == 'object':

                    for obj_elem in elem:
                        if obj_elem.tag == 'bndbox':
                            line = []
                            for bbox_elem in obj_elem:
                                value = float(bbox_elem.text)
                                line.append(value)
                            boxes.append(line)

            if image_path is not None and boxes != []:
                
                self.image_paths.append(image_path)
                self.boxes.append(boxes)


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images_with_label',self.image_paths[idx])
        boxes = self.boxes[idx]
        boxes = torch.FloatTensor(boxes)

        img = Image.open(img_path).convert('RGB')
        print(img)
        if self.transform:
            
            #print('Old width: ' + str(img.width))
            old = img.width
            img = self.transform(img)
            #print('New width: ' + str(img.shape))
            factor = old / img.shape[1]
            #print('Old box:' + str(boxes))
            #print('Factor:' + str(factor))
            
            boxes = list(map(lambda l: list(map(lambda v:v/factor,l)), boxes))
            #print('New box: ' + str(boxes))

        return img, boxes
    
    def show_image(self, idx):
        img,_ = self.__getitem__(idx)
        
        
        plt.imshow(img.permute(1,2,0))
    
'''
EXEMPLE OF USAGE:

ROOT = os.getcwd()
DATASET = ROOT + "\\dataset"
dataset = PoolDataset(root_dir=DATASET)
img, label = dataset[5]

'''