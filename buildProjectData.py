# This file pretends to merge the datasets that we have letting all the data organized, ignoring the Algarve one for now!

import os
import shutil

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
CANNES_IMAGES = ROOT + "\\Cannes\\images"
CANNES_LABELS = ROOT + "\\Cannes\\labels"
SWIMMING_POOL_TRAINING_IMAGES = ROOT + "\\swimmingPool\\training\\images"
SWIMMING_POOL_TRAINING_LABELS = ROOT + "\\swimmingPool\\training\\labels"
SWIMMING_POOL_TESTING_IMAGES = ROOT + "\\swimmingPool\\testing\\images"
SWIMMING_POOL_TESTING_LABELS = ROOT + "\\swimmingPool\\testing\\labels"

DATASET_IMAGES_LABELED = ROOT + "\\dataset\\images_with_label"
DATASET_IMAGES_NOT_LABELED = ROOT + "\\dataset\\images_without_labels"
DATASET_LABELS = ROOT + "\\dataset\\labels"

print("Clear the destination folders")
delete_files_from_dir(DATASET_IMAGES_LABELED)
delete_files_from_dir(DATASET_IMAGES_NOT_LABELED)
delete_files_from_dir(DATASET_LABELS)
print()

list_cannes_images = os.listdir(CANNES_IMAGES)
list_cannes_labels = os.listdir(CANNES_LABELS)

list_sp_training_images = os.listdir(SWIMMING_POOL_TRAINING_IMAGES)
list_sp_testing_images = os.listdir(SWIMMING_POOL_TESTING_IMAGES)
list_sp_training_labels = os.listdir(SWIMMING_POOL_TRAINING_LABELS)
list_sp_testing_labels = os.listdir(SWIMMING_POOL_TESTING_LABELS)

print("The Cannes dataset has %s images" % len(list_cannes_images))
print("And %s are labeled" % len(list_cannes_labels))
dif_cannes = len(list_cannes_images) - len(list_cannes_labels)

print()

print("The swimming pool dataset has %s images" % (len(list_sp_testing_images) + len(list_sp_training_images)))
print("And %s are labeled" % (len(list_sp_training_labels) + len(list_sp_testing_labels)))
dif_sp = len(list_sp_testing_images) + len(list_sp_training_images) - len(list_sp_training_labels) - len(list_sp_testing_labels)

# Cannes
cannes_labels_names = list(map(lambda v: v.split(".")[1], list_cannes_labels))
c = 0 # confirmation - checked
for val in list_cannes_images:
    src  = CANNES_IMAGES + "\\" + val
    if val.split(".")[1] in cannes_labels_names:
        shutil.copy(src, DATASET_IMAGES_LABELED + "\\" + val)

        new_val = ".".join(val.split(".")[0:2]) + ".xml"
        src_2 = CANNES_LABELS + "\\" + new_val
        shutil.copy(src_2, DATASET_LABELS + "\\" + new_val)

    else:
        # c+=1
        shutil.copy(src, DATASET_IMAGES_NOT_LABELED + "\\" + val)


# Swimming Pool - it will be necessary to rename them to merge
# it is better to do it in 2 steps, one for training and one for testing
prefix = "sp_"
counter = 0

# for training
sp_training_labels_names = list(map(lambda v: v.split(".")[0], list_sp_training_labels))

for val in list_sp_training_images:
    src  = SWIMMING_POOL_TRAINING_IMAGES + "\\" + val
    new_name = prefix + str(counter)
    if val.split(".")[0] in sp_training_labels_names:
        shutil.copy(src, DATASET_IMAGES_LABELED + "\\" + new_name + ".jpg")

        src_2 = SWIMMING_POOL_TRAINING_LABELS + "\\" + val.split(".")[0] + ".xml"
        shutil.copy(src_2, DATASET_LABELS + "\\" + new_name + ".xml")

    else:
        # c+=1
        shutil.copy(src, DATASET_IMAGES_NOT_LABELED + "\\" + new_name + ".jpg")

    counter +=1

# print(counter)

# for testing

sp_testing_labels_names = list(map(lambda v: v.split(".")[0], list_sp_testing_labels))

for val in list_sp_testing_images:
    src  = SWIMMING_POOL_TESTING_IMAGES + "\\" + val
    new_name = prefix + str(counter)
    if val.split(".")[0] in sp_testing_labels_names:
        shutil.copy(src, DATASET_IMAGES_LABELED + "\\" + new_name + ".jpg")

        src_2 = SWIMMING_POOL_TESTING_LABELS + "\\" + val.split(".")[0] + ".xml"
        shutil.copy(src_2, DATASET_LABELS + "\\" + new_name + ".xml")

    else:
        # c+=1
        shutil.copy(src, DATASET_IMAGES_NOT_LABELED + "\\" + new_name + ".jpg")

    counter +=1

# Confirmations 
sil = len(list_cannes_labels) + len(list_sp_training_labels) + len(list_sp_testing_labels)
sinl = dif_cannes + dif_sp

print("---- CONFIRMATIONS ----")
print("We should have %d images labeled, check = %s" % (sil, len(os.listdir(DATASET_IMAGES_LABELED))== sil))
print("We should have %d images without labels, check = %s" % (sinl, len(os.listdir(DATASET_IMAGES_NOT_LABELED))== sinl))
print("We should habe %d labels, check = %s" % (sil, len(os.listdir(DATASET_LABELS))== sil))






