"""
Created on Fri Mar 30 12:39:26 2018
This is the demo code. That should run without making any changes.
Please ensure that demoImage.hdf5 is in the same directory as this file tstDemo.py.

This code will load the learned model from the subdirectory 'savedModels'

This test code will load an  image for  from the demoImage.hdf5 file.

@author: haggarwal
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import supportingFunctions as sf
import model as mm
import scipy.io as sio
import h5py as h5py
from tqdm import tqdm
cwd=os.getcwd()


cwd=os.getcwd()
tf.reset_default_graph()

#%% choose a model from savedModels directory
nLayers=5
epochs=50
gradientMethod='AG'
K=15
sigma=0.00
#subDirectory='14Mar_1105pm'
subDirectory='16Nov_0349pm_5L_15K_50E_AG'
#%%Read the testing data from dataset.hdf5 file

#tstOrg is the original ground truth
#tstAtb: it is the aliased/noisy image
#tstCsm: this is coil sensitivity maps
#tstMask: it is the undersampling mask

tstOrg,tstAtb,tstCsm,tstMask=sf.getTestingData()
batchSize = 1
#you can also read more testing data from dataset.hdf5 (see readme) file using the command
#tstOrg,tstAtb,tstCsm,tstMask=sf.getData('testing',num=100)

#%% Load existing model. Then do the reconstruction
nTst=tstOrg.shape[0]
nBatch= int(np.floor(np.float32(nTst)/batchSize))
nSteps= nBatch

print ('Now loading the model ...')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

modelDir= cwd+'/savedModels/'+subDirectory #complete path
rec=np.empty(tstAtb.shape,dtype=np.complex64) #rec variable will have output

tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

csmT = tf.placeholder(tf.complex64,shape=(None,1,320,320),name='csmT')
maskT= tf.placeholder(tf.complex64,shape=(None,320,320),name='maskT')
atbT = tf.placeholder(tf.float32,shape=(None,320,320,2),name='atbT')
#orgT = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')
out=mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod)
predT=out['dc'+str(K)]
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
with tf.Session(config=config) as sess:
#    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    saver.restore(sess,loadChkPoint)
#    graph = tf.get_default_graph()
#    predT = tf.placeholder(tf.float32,shape=(None,320,320,2),name='predT')
#    predT = tf.placeholder(tf.float32,shape=(None,320,320,2),name='predT')
#    predT =graph.get_tensor_by_name('predTst:0')
#    maskT =graph.get_tensor_by_name('mask:0')
#    atbT=graph.get_tensor_by_name('atb:0')
#    csmT   =graph.get_tensor_by_name('csm:0')
#    csmT = tf.placeholder(tf.complex64,shape=(None,None,None,None),name='csmT')
#    maskT= tf.placeholder(tf.complex64,shape=(1,320,320),name='maskT')
#    atbT = tf.placeholder(tf.float32,shape=(None,320,320,2),name='atbT')
    for step in tqdm(range(nSteps)):
        dataDict={atbT:tstAtb[step].reshape(-1,320,320,2),maskT:tstMask[step].reshape(-1,320,320),csmT:tstCsm[step].reshape(-1,1,320,320) }
        rec[step]=sess.run(predT,feed_dict=dataDict)
rec=sf.r2c(rec.squeeze())
print('Reconstruction done')

#%% normalize the data for calculating PSNR

print('Now calculating the PSNR (dB) values')

normOrg=sf.normalize01( np.abs(tstOrg))
normAtb=sf.normalize01( np.abs(sf.r2c(tstAtb)))
normRec=sf.normalize01(np.abs(rec))

psnrAtb=sf.myPSNR(normOrg,normAtb)
psnrRec=sf.myPSNR(normOrg,normRec)

print ('*****************')
print ('  ' + 'Noisy ' + 'Recon')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb,psnrRec))
print ('*****************')

#%% Display the output images
#plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
#plt.clf()
#plt.subplot(141)
#plot(np.fft.fftshift(tstMask[0]))
#plt.axis('off')
#plt.title('Mask')
#plt.subplot(142)
#plot(normOrg)
#plt.axis('off')
#plt.title('Original')
#plt.subplot(143)
#plot(normAtb)
#plt.title('Input, PSNR='+str(psnrAtb.round(2))+' dB' )
#plt.axis('off')
#plt.subplot(144)
#plot(normRec)
#plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
#plt.axis('off')
#plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
#plt.show()


#a_dict = {'field1':tstOrg, 'field2': rec}
#sio.savemat('recon50_9.mat', {'a_dict': a_dict})
#path = './recons_h5/'
#savename = path+ 'recon_pat13V2_50.h5'
#hf = h5py.File(savename, 'w')
##hf.create_dataset('field1', data=tstOrg)
#hf.create_dataset('field2', data=rec)
#hf.close()
