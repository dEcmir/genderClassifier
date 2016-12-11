
import numpy as np
import sys
import os
import h5py

CAFFE_PATH = "/home/pierre/caffe/python/"
sys.path.append(CAFFE_PATH)
import caffe

direc = 'vgg_face_caffe/'
image_dir = 'img_align_celeba/'

imagespaths = sorted(os.listdir(image_dir))


model = direc+'VGG_FACE_deploy.prototxt'
weights = direc+'VGG_FACE.caffemodel'
caffe.set_mode_gpu();
net = caffe.Net(model, weights, caffe.TEST)  # create net and load weights

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(
    CAFFE_PATH +"caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(100,3,224,224)

#load the image in the data layer
#f = h5py.File('celebfeatures.hdf5','w')

#dset = f.create_dataset("features",
#                        (len(imagespaths),4096),
#                        dtype='float32')
batch = np.empty((100,3,224,224))
for i in range(len(imagespaths)):
    im = caffe.io.load_image(image_dir+imagespaths[i])
    net.blobs['data'].data[i % 100] = transformer.preprocess('data', im)
    if (i % 100 == 99) or (i == len(imagespaths)-1):
        net.forward()
        if i % 100 == 99:
            dset[i-99:i+1] = net.blobs['fc7'].data
        else:
            import pdb; pdb.set_trace()
            dset[i-(i%100):] = net.blobs['fc7'].data[:i%100+1]

sex = np.loadtxt('list_attr_celeba.txt', skiprows=2, usecols=(21,)).astype('i8')
dset = f.create_dataset("sex", sex)
f.flush()
f.close()







prob = out['prob']
