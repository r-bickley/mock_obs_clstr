from six.moves import urllib
from sklearn.decomposition import PCA
from scipy.io import loadmat
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing
import keras
from IPython.display import clear_output
from sklearn.linear_model import LogisticRegression
from keras.layers import Input, Dense
from keras.models import Model
import itertools
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tarfile
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras import backend as K
import os, sys
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import cv2
from skimage.transform import resize
from sklearn.manifold import TSNE

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="train")
        plt.plot(self.x, self.val_losses, label="validation",linestyle='--')
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

def im_look(filename,cam):
    sci_tar = tarfile.open(filename)
    membs=sci_tar.getmembers()
    im_dat = fits.getdata(sci_tar.extractfile(membs[cam]))
    sci_tar.close()
    return im_dat

#Retrieve galaxy image data
data_dir = '/home/rbickley/projects/rrg-jfncc/rbickley/ms-ill-cnn/sci_ims_1/'
data_files = os.listdir(data_dir)[:10000]
print(len(data_files))
print(data_files[0])
data_files = [data_dir+i for i in data_files]
inp = np.array([resize(im_look(f,c),(128,128)) for f in data_files for c in range(4)])
print('data imported')

#take naive "artifact" statistic of f_zer
f_zers = [len(np.argwhere(im>1))/(len(im)**2) for im in inp]
#normalize
inp = [i-np.amin(i) for i in inp]
inp = [i/np.amax(i) for i in inp]
inp_flat = np.reshape(inp,(-1,16384))

inp_tr,inp_va = train_test_split(inp_flat,test_size=.2,random_state=0)
fr_tr,fr_va = train_test_split(f_zers,test_size=.2,random_state=0)
print('normalized & fractions calculated')

#Autoencoder CNN
input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
z = MaxPooling2D((2, 2), padding='same', name='latent_layer')(x)
z= keras.layers.BatchNormalization()(z)

x = Conv2D(4, (3, 3), activation='relu', padding='same')(z)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# this model maps an input to its encoded representation
encoder = Model(input_img, z)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.summary()

#separate encoder-only model
encoder = Model(input_img,z)
encoder.summary()

inp_tr2D = np.reshape(inp_tr, (len(inp_tr), 128, 128, 1))  # adapt this if using `channels_first` image data format
inp_va2D = np.reshape(inp_va, (len(inp_va), 128, 128, 1))  # adapt this if using `channels_first` image data format
print('encoder defined & data prepared')

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.fit(inp_tr2D, inp_tr2D,
#                epochs=100,
#                batch_size=128,
#                shuffle=True,
#                validation_data=(inp_va2D, inp_va2D))
#autoencoder.save_weights('autoenc_model_final.h5')
autoencoder.load_weights('autoenc_model_final.h5')
print('trained, weights saved')

encoded_va = encoder.predict(inp_va2D)
encoded_tr = encoder.predict(inp_tr2D)

# predict
from sklearn.cluster import KMeans
##Choose the number of clusters  in the Kmean method
K = 7
# fit the n first components of pca  by Kmean 
kmeans = KMeans(n_clusters=K).fit(encoded_tr.reshape(32000,-1))
Kmean_tr=kmeans.predict(encoded_tr.reshape(32000,-1))
Kmean_va=kmeans.predict(encoded_va.reshape(8000,-1))

comp_x = 0
comp_y = 1
colors = ['b','g','r','c','m','y','black']
fig1 = plt.figure(figsize=[10,10])
for k1 in range(K):
    plt.scatter(kmeans.cluster_centers_[k1,comp_x],kmeans.cluster_centers_[k1,comp_y],s=300,c=colors[k1],marker='o')
    plt.scatter(encoded_va.reshape(8000,-1)[Kmean_va==k1][comp_x],encoded_va.reshape(8000,-1)[Kmean_va==k1][comp_y],c=colors[k1],alpha=0.1)
plt.savefig('raw_kmeans.png')

tsne = TSNE(n_components=2)
tsne.fit(encoded_tr.reshape(32000,-1))
tsne_results_tr= tsne.fit_transform(encoded_tr.reshape(32000,-1))

#t-SNE plot
comp_x=0
comp_y=1

plt.figure(figsize=(12,8))
plt.scatter(tsne_results_tr[:,comp_x], tsne_results_tr[:,comp_y], edgecolor='none', alpha=0.5,c=Kmean_tr,cmap=plt.cm.get_cmap('nipy_spectral',10))
plt.xlabel('component_ '+ str(comp_x))
plt.ylabel('component_ '+str(comp_y))
plt.colorbar()
plt.savefig('t-SNE.png')

#alt t-SNE plot to show separation of naively-defined artifacts
plt.figure(figsize=(12,8))
plt.scatter(tsne_results_tr[:,comp_x], tsne_results_tr[:,comp_y], edgecolor='none', alpha=0.5,c=fr_tr,cmap='viridis')
plt.xlabel('component_ '+ str(comp_x))
plt.ylabel('component_ '+str(comp_y))
plt.colorbar()
plt.savefig('t-SNE-fr.png')

fig, axs = plt.subplots(2,10,figsize=(20, 4))
count = 0
for i in range(2):
    for j in range(10):
        plt.gray()
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==0][count],(128,128))))
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_0.png')

fig, axs = plt.subplots(2,10,figsize=(20, 4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==1][count],(128,128))))
        plt.gray()
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_1.png')

fig, axs = plt.subplots(2,10,figsize=(20, 4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==2][count],(128,128))))
        plt.gray()
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_2.png')

fig, axs = plt.subplots(2,10,figsize=(20, 4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==3][count],(128,128))))
        plt.gray()
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_3.png')

fig, axs = plt.subplots(2,10,figsize=(20, 4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==4][count],(128,128))))
        plt.gray()
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_4.png')

fig, axs = plt.subplots(2,10,figsize=(20,4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==5][count],(128,128))))
        plt.gray()
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)
        count +=1
plt.savefig('category_5.png')

fig, axs = plt.subplots(2,10,figsize=(20,4))
count = 0
for i in range(2):
    for j in range(10):
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==6][count],(128,128))))
        plt.gray()
        axs[i][j].imshow(np.log10(np.reshape(inp_tr[Kmean_tr==5][count],(128,128))))
        axs[i][j].get_yaxis().set_visible(False)
        count += 1
plt.savefig('category_6.png')
