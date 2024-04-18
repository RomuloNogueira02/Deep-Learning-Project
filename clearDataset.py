# Created this function, to dont fill GitHub with images
import os


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


ROOT = os.getcwd()

DATASET_IMAGES_LABELED = ROOT + "\\dataset\\images_with_label"
DATASET_IMAGES_NOT_LABELED = ROOT + "\\dataset\\images_without_labels"
DATASET_LABELS = ROOT + "\\dataset\\labels"

delete_files_from_dir(DATASET_IMAGES_LABELED)
delete_files_from_dir(DATASET_IMAGES_NOT_LABELED)
delete_files_from_dir(DATASET_LABELS)