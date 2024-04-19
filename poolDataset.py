import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch

class PoolDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.parse_xml()
    
    def parse_xml(self):
        xml_files = [file for file in os.listdir(os.path.join(self.root_dir, 'labels')) if file.endswith('.xml')]

        for xml_file in xml_files:
            tree = ET.parse(os.path.join(self.root_dir, 'labels',xml_file))
            root = tree.getroot()

            image_path = None
            label = []

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
                            label.append(line)

            if image_path is not None and label != []:
                
                self.image_paths.append(image_path)
                self.labels.append(label)


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images_with_label',self.image_paths[idx])
        label = self.labels[idx]
        label = torch.FloatTensor(label)

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img, label = self.transform(img, label)

        return img, label
    
'''
EXEMPLE OF USAGE:

ROOT = os.getcwd()
DATASET = ROOT + "\\dataset"
dataset = PoolDataset(root_dir=DATASET)
img, label = dataset[5]

'''