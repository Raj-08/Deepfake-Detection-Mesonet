import numpy as np
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
import tensorflow as tf
import sys
import os
sys.path.append('/floyd/home/Model/')
print(os.getcwd())
from model_contour import build_model
from utils import random_crop_and_pad_image_and_labels

import cv2
import time
import cv2
import numpy as np
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
from face_detection.face_detection import RetinaFace
from matplotlib import pyplot as plt
#import face_recognition
import os
import skimage.io
import numpy as np
import cv2
detector = RetinaFace(0)
from tensorflow.python.ops import variables
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS
import random
def save(saver, sess, logdir, step):

   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
       
def main(unused_argv):    
    
    image_ph = tf.placeholder(tf.float32,[1,256,256,3],name='image_placeholder')
    pred_sigmoid,pred = build_model(image_ph)
    loader = tf.train.Saver(var_list=tf.global_variables() )
    init = variables.global_variables_initializer()
    folders='/floyd/home/tc/' ##your folder containing fake videos to test
    count_true=0
    count_false=0
    with tf.Session() as sess:
        sess.run(init)
        #if FLAGS.tf_initial_checkpoint==True:
        load(loader, sess, '/floyd/home/trueaware_checkpoint/model.ckpt-464000')
        print('Training Starts........')
        step_iter = 0
        alls=os.listdir(folders)
        for al in alls[0:1]:
            cap = cv2.VideoCapture(folders+al)
            print(folders+al)
            property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
            length = int(cv2.VideoCapture.get(cap, property_id))
            #frame= random.randint(0,length)
            count=0
            for frame in range(0,length):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
                    res, frame = cap.read()
                    frame_or=frame.copy()
                    faces = detector(frame)
                    box, landmarks, score = faces[0]
                    box = box.astype(np.int)
                    fimg=frame[box[1]-10:box[3]+10,box[0]-10:box[2]+10,:]
                    faces = cv2.cvtColor(np.asarray(fimg), cv2.COLOR_BGR2RGB)
                    frame_or = cv2.cvtColor(np.asarray(frame_or), cv2.COLOR_BGR2RGB)
                    faces_or=faces.copy()
                    faces=cv2.resize(faces,(256,256),interpolation=cv2.INTER_CUBIC)
                    input_image = faces.copy()
                    plt.imshow(input_image)
                    plt.show()
                    input_image = np.expand_dims(faces,axis=0)
                    start_time = time.time()

                    feed_dict={image_ph:input_image}#,label_ph:class_id}
                    P= sess.run(pred_sigmoid, feed_dict=feed_dict)

                    cv2.rectangle(frame_or,(box[0]-10,box[1]-10),(box[2]+10,box[3]+10),(0,255,0),2)
                    cv2.putText(frame_or,'DEEPFAKE_DETECTED    SCORE  ::  '+str(P[0][0]),(box[0],box[1]-20),0,0.2,(255,0,0))
                    skimage.io.imsave('/floyd/home/test_res/'+str(count)+'.jpg',frame_or)
                    print(P[0][0])
                    count=count+1
                    if P[0][0]>0.65:
                        print('DEEPFAKE DETECTED')
                        count_true=count_true+1

                    else:
                        print('DEEPFAKE NOT DETECTED')
                        count_false=count_false+1
        
                except:
                    print("ERROR")

if __name__ == '__main__':
  tf.app.run()
