import numpy as np
import tensorflow as tf
from model_deepfake import build_model
from utils import random_crop_and_pad_image_and_labels
import os
import cv2
import time
import cv2
import numpy as np
from face_detection.face_detection import RetinaFace
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
flags.DEFINE_integer('max_to_keep', 50,
                     'Maximium number of checkpoints to be saved.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('Epochs', 100,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_integer('train_crop_size', 256 ,
                           'Image crop size [height, width] during training.')

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

flags.DEFINE_float('learning_rate', .0000001,
                   'Learning rate employed during slow start.')

flags.DEFINE_string('image_dir', None,
                    'The Image Directory.')

flags.DEFINE_string('label_dir', None,
                    'The Label Directory.')

flags.DEFINE_string('log_dir', None,
                    'The Logs Directory.')

flags.DEFINE_float('clip_by_value', 1.0, 'The value to be used for clipping.')

 
flags.DEFINE_string('train_text', None,
                    'The Path to the text file containing names of Images and Labels')###This text file should not have extensions in their names such as 8192.png or 8192.jpg instead just the name such as 8192

Image_directory = FLAGS.image_dir
Label_directory = FLAGS.label_dir
my_log_dir = FLAGS.log_dir

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
    print(pred)
    loader = tf.train.Saver(var_list=tf.global_variables() )
    init = variables.global_variables_initializer()
    folders='/home/ubuntu/Trueaware/Test_Fake_Videos/'
    count_true=0
    count_false=0
    with tf.Session() as sess:
        sess.run(init)
        #if FLAGS.tf_initial_checkpoint==True:
        load(loader, sess, './checkpoint/model.ckpt-700000')
        print('Training Starts........')
        step_iter = 0
        alls=os.listdir(folders)
        for al in alls:
            try :
                cap = cv2.VideoCapture(folders+al)
                print(folders+al)
                property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
                length = int(cv2.VideoCapture.get(cap, property_id))

                frame= random.randint(0,length)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
                res, frame = cap.read()
                faces = detector(frame)
                box, landmarks, score = faces[0]
                box = box.astype(np.int)
                fimg=frame[box[1]-10:box[3]+10,box[0]-10:box[2]+10,:]
                faces = cv2.cvtColor(np.asarray(fimg), cv2.COLOR_BGR2RGB)
                faces=cv2.resize(faces,(256,256),interpolation=cv2.INTER_CUBIC)
                input_image = faces.copy()
                input_image = np.expand_dims(faces,axis=0)
                start_time = time.time()

                feed_dict={image_ph:input_image}#,label_ph:class_id}
                P= sess.run(pred_sigmoid, feed_dict=feed_dict)
                print(P[0][0])

                if P[0][0]>0.65:    ## Threshold 
                    print('DEEPFAKE DETECTED')
                    count_true=count_true+1
                    
                else:
                    print('DEEPFAKE NOT DETECTED')
                    count_false=count_false+1
        
            except:
                print("ERROR")
    print('TRUE DETECTION ::',count_true)
    print('FALSE DETECTION ::',count_false)

if __name__ == '__main__':
  tf.app.run()
