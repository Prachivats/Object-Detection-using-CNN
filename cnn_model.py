import matplotlib.image as mpimg
import matplotlib.pyplot as mp
import glob
from cnn_network import *
#from skimage import io
import numpy as np
from skimage.transform import rescale, resize


print('Loading and Preparing Training Image data......')

# Loading the Training Images data
train_img = []
training_img_dir = r"/home/ajay/kaam/CNN/training_datasets"
for img in glob.glob('/home/ajay/kaam/CNN/training_datasets/*'+'/*.jpg'):
    #     cv_img.append(resize(io.imread(img, as_grey = True)/255, (100, 100)))
    # Image Preprocessing
    train_img.append(resize(mpimg.imread(img)/255, (100, 100, 3)))
    train_imgs = np.array(train_img)
train_imgs -= int(np.mean(train_imgs))

# Creating the Training Images Class Labels with One hot Coding
Class_num = 6
class_lab = []
classes = glob.glob('/home/ajay/kaam/CNN/training_datasets/*')
count = 0
for clas in classes:
    c_len = len(glob.glob(clas + '/*.jpg'))
    class_lab.extend([count] * c_len)
    count += 1
training_labels = np.eye(Class_num)[class_lab]
print (training_labels.shape)
print(" ### train_shape", train_imgs.shape)
batch_size = 32
Epochs = 1
cnn = cnn_network()
print('Training CNN......')
#  Training the CNN Model
cnn.training(train_imgs, training_labels, batch_size, Epochs)
