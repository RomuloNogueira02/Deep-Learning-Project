import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


DATASET_IMAGES_LABELED = "\\dataset\\images_with_label"
DATASET_LABELS =  "\\dataset\\labels"

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
            label = None

            for elem in root:

                if elem.tag == 'path':
                    image_path = elem.text

                if elem.tag == 'object':
                    label = elem.text

            if image_path is not None and label is not None:
                
                self.image_paths.append(image_path)
                self.labels.append(label)
                

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images_with_label',self.image_paths[idx])
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
