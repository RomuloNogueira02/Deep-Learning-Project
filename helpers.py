import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def show_image(image):
    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    plt.show()

    
def show_image_with_box(image,boxes):

    image_draw = image.permute(1,2,0).clone().cpu().numpy()
    boxes_np = [box.tolist() for box in boxes]

    for box in boxes_np:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    
    plt.imshow(image_draw)
    plt.axis('off')
    plt.show()

def get_images_labeled(labels_path):
    return os.listdir(labels_path)

def change_image_to_label(image_path):
    new_path = image_path.replace('images','labels')
    if 'jpg' in new_path:
        new_path = new_path.replace('jpg','xml')
    elif 'png' in new_path:
        new_path = new_path.replace('png','xml')
    
    return new_path

def getBoxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for elem in root:
        if elem.tag == 'object':
            for obj_elem in elem:
                if obj_elem.tag == 'bndbox':
                    line = []
                    for bbox_elem in obj_elem:
                        value = float(bbox_elem.text)
                        line.append(value)
                    boxes.append(line)
    return boxes

def getLabels(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []

    for elem in root:
        if elem.tag == 'object':
            for obj_elem in elem:
                if obj_elem.tag == 'name':
                    labels.append(1)
    return labels

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def delete_files_from_dir(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully from %s " % (directory_path))
   except OSError:
     print("Error occurred while deleting files.")