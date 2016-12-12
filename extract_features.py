
import numpy as np
import sys
import os
import h5py


def main(caffe_path, image_dir, ann_file, max_number=None):
    caffe_path = "/home/pierre/caffe/python/"
    sys.path.append(caffe_path)
    import caffe

    direc = 'vgg_face_caffe/'


    imagespaths = sorted(os.listdir(image_dir))
    if max_number is not None:
        imagespaths = imagespaths[:max_number]

    # create caffe model
    model = direc+'VGG_FACE_deploy.prototxt'
    weights = direc+'VGG_FACE.caffemodel'
    caffe.set_mode_gpu();
    net = caffe.Net(model, weights, caffe.TEST)  # create net and load weights

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(
        caffe_path +"caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(100,3,224,224)

    # create hdf5 database
    f = h5py.File('celebfeatures.hdf5','w')

    dset = f.create_dataset("features",
                            (len(imagespaths),4096),
                            dtype='float32')
    # batch based feature computation, this can take some time
    batch = np.empty((100,3,224,224))
    for i in range(len(imagespaths)):
        im = caffe.io.load_image(image_dir+imagespaths[i])
        net.blobs['data'].data[i % 100] = transformer.preprocess('data', im)
        if (i % 100 == 99) or (i == len(imagespaths)-1):
            net.forward()
            if i % 100 == 99:
                dset[i-99:i+1] = net.blobs['fc7'].data
            else:
                dset[i-(i%100):] = net.blobs['fc7'].data[:i%100+1]

    sex = np.loadtxt(ann_file, skiprows=2, usecols=(21,)).astype('i8')
    if max_number is not None:
        sex = sex[:max_number]
    dset = f.create_dataset("sex", data=sex)
    f.flush()
    f.close()

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print "Usage :", sys.argv[0], "<pycaffe path> <img_align_celeba path>",
        print "<list_attr_celeba.txt path> [max_number]"
    else:
        caffe_path = sys.argv[1]
        image_dir = sys.argv[2]
        ann_file = sys.argv[3]
        if len(sys.argv)>4:
            max_number = int(sys.argv[4])
        else:
            max_number = None
        main(caffe_path, image_dir, ann_file, max_number=max_number)
