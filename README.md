# Tensorflow-Deepfake-Detection-Mesonet
A Tensorflow Implementation of MesoNet: a Compact Facial Video Forgery Detection Network (https://arxiv.org/abs/1809.00888)


**REQUIREMENTS :**

```
pip install requirements.txt
```
**Label Preparation :**

To prepare the labels for contour detection from PASCAL Dataset , run create_lables.py and edit the file to add the path of the labels and new labels to be generated . Use this path for labels during training. 

**TRAINING :**

```
python train.py \
    --max_to_keep=50 \
    --Epochs=100 \
    --momentum=0.9 \
    --learning_rate=.0000001 \
    --train_crop_size=480 \
    --clip_by_value=1.0 \
    --train_text = ${path to text file} \
    --log_dir = ${path to where logs will be saved} \
    --tf_initial_checkpoint=${PATH_TO_CHECKPOINT} \
    --label_dir = ${path to label directory} \
    --image_dir = ${path to image directory}
```
**EVALUATION :**
```
python eval.py \
    --checkpoint=${path to checkpoint to be evaluated} \
    --save_preds=${path to folder where predictions will be saved} \
    --image_dir = ${path to image directory} \
    --eval_crop_size=480 \
    --eval_text =  ${path to eval text file}

```
**Results :**

<img src="./images (36).jpeg" alt="Image_1"/>

<!-- <img src="./000999.png" alt="prediction_1"/>

<img src="./000129.jpg" alt="Image_1"/>

<img src="./000129.png" alt="prediction_1"/>
 -->
