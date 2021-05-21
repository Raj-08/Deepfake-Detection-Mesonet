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
    label_ph = tf.placeholder(tf.uint8,[1,1],name='label_placeholder')
    size = FLAGS.train_crop_size


    pred_sigmoid,pred = build_model(image_ph)
    loader = tf.train.Saver(var_list=tf.global_variables() )
    total_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.squeeze(label_ph),dtype=tf.float32),logits =tf.squeeze(pred))
    total_loss = tf.reduce_sum(total_loss)
    all_trainables = tf.trainable_variables()

    total_loss_scalar = tf.summary.scalar("total_cost", total_loss)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
    train_summary_op = tf.summary.merge([total_loss_scalar])
    train_writer = tf.summary.FileWriter('./train',
                                        graph=tf.get_default_graph())
    optimizer=tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')
    grads = tf.gradients(total_loss, all_trainables)
    grads_and_vars = zip(grads, all_trainables)              
    train_op = optimizer.apply_gradients(grads_and_vars)
    
    init = variables.global_variables_initializer()
    folders=['Real_Videos5','Real_Videos4','Real_Videos3','Real_Videos2','Real_Videos1','Fake_Videos4','Fake_Videos3','Fake_Videos2','Fake_Videos1','Fake_Videos5','Fake_Videos6','Fake_Videos7','Fake_Videos8','Fake_Videos9','Fake_Videos10']
    with tf.Session() as sess:
        sess.run(init)
        #if FLAGS.tf_initial_checkpoint==True:
        load(loader, sess,'./checkpoint/model.ckpt-464000')
        print('Training Starts........')
        step_iter = 464000
        for epoch in range(10000000):
            if step_iter%4000==0:
                save(saver, sess, './checkpoint/', step_iter)
            i=0;
            rand_fold=random.choice(folders)
            all_videos=os.listdir('/home/ubuntu/Trueaware/Real_And_Fake_Videos/'+rand_fold)
            rand_video=random.choice(all_videos)
            try :
                step_iter =step_iter+1
                i=i+1;
                cap = cv2.VideoCapture('/home/ubuntu/Trueaware/Real_And_Fake_Videos/'+rand_fold+'/'+rand_video)
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
                if 'Real' in rand_fold:
                    class_id=[[0]]
                elif 'Fake' in rand_fold:
                    class_id=[[1]]
                feed_dict={image_ph:input_image,label_ph:class_id}
                L,P,_,sum_op = sess.run([total_loss,pred_sigmoid,train_op,train_summary_op], feed_dict=feed_dict)
                if 'Real' in rand_fold:
                    skimage.io.imsave('./real.jpg',faces)
                elif 'Fake' in rand_fold:
                    skimage.io.imsave('./fake.jpg',faces)
                train_writer.add_summary(sum_op, step_iter)     
                duration = time.time() - start_time
                print('::Step::'+str(epoch)+','+str(i), '::total_loss::'+ str(L),'::time::'+str(duration))
            except:
                print("ERROR")

if __name__ == '__main__':
  tf.app.run()
