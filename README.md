![Example of the different model outputs](/Images/examples.png)

# Model comparison for Pool Detection in aerial images

## Abstract:
Standard object detection methods are widely used for
various tasks. Meanwhile, with the appearance of transformers,
their use for object detection has become more
and more popular. We study this new aproach, by experimenting
and comparing the different methods. We use preimplemented
versions of the various algorithms and train
them for this specific task. We experiment on a custom
dataset which results from merging two different datasets
publicly available on the Kaggle platform.

## Main contributions
- Provide a comprehensive pipeline to train three different
models for this task.
- Comparison of traditional models with transformers
based models.
- Scripts that can be used to extract principal features from
a raw dataset (image, label) attending specific features of
each model.

## Models Trained
- Faster R_CNN
- YOLO
- DETR

## Results:

![Results of the different models](/Images/results.png)

## Repo description:
* **/DETR**
   * How we loaded the dataset into DETR
   * How we trained DETR
* **/Faster R_CNN**
   * Custom made Dataset Class
   * How we trained Faster R-CNN
* **/YOLO**
   * Data Treatment for YOLO requirements
   * How we trained YOLO
   * /runs
      * YOLO run files   
  
## Authors:
* Guilherme Ribeiro
    * fc53699@alunos.fc.ul.pt
* Rómulo Nogueira
    * fc56935@alunos.fc.ul.pt

## Note:
This project was developed during the Deep Learning class taught by Prof. Nuno Garcia at the Faculty of Science, University of Lisbon.
