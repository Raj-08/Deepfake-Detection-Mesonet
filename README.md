# Tensorflow-Deepfake-Detection-Mesonet
A Tensorflow Implementation of MesoNet: a Compact Facial Video Forgery Detection Network (https://arxiv.org/abs/1809.00888)


**REQUIREMENTS :**

Mesonet utilises Deepfake detection on Cropped Faces. For this a fast and accurate pytorch based tool is used to crop the faces. 

```
git clone https://github.com/elliottzheng/face-detection
```
**Label Preparation :**

To prepare the labels for contour detection from PASCAL Dataset , run create_lables.py and edit the file to add the path of the labels and new labels to be generated . Use this path for labels during training. 

**TRAINING :**

```
python train.py \

```

**DATASET**
```

We use the Deepfake dataset available publicly at Facebook. (https://ai.facebook.com/datasets/dfdc/)

```
**Results :**

<img src="./Disinf-GIF.gif" alt="Image_1"/>

<!-- <img src="./000999.png" alt="prediction_1"/>

<img src="./000129.jpg" alt="Image_1"/>

<img src="./000129.png" alt="prediction_1"/>
 -->
