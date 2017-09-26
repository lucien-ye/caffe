import sys
import os

import numpy as np 
import os.path as osp
caffe_root = '../'

sys.path.append(caffe_root + 'python')
sys.path.append("pycaffe") # the tools file is in this folder
sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import matplotlib.pyplot as plt

from copy import copy

# plt.rcParams['figure.figsize'] = (6, 6)


import tools
#pascal_root = osp.join(caffe_root, 'data/VOC2012/VOCdevkit/VOC2012/')
pascal_root = osp.join('/home/dailingzheng/caffe/data/VOC2012/VOCdevkit/VOC2012/')


classes = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downlaod pre-trained CaffeNet model ...")

    caffe.set_mode_cpu()


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper



workdir = './MobileNet_pascal_classify'
if not os.path.isdir(workdir):
    os.makedirs(workdir)

solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), 
                                         testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
solverprototxt.sp['display'] = "10"
solverprototxt.sp['base_lr'] = "0.0001"
solverprototxt.write(osp.join(workdir, 'solver.prototxt'))



solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(caffe_root + 'models/mobilenet/mobilenet.caffemodel')

#define the Net 
# caffe.set_mode_cpu
# model_def = caffe_root + '../../deploy.prototxt'
# model_weight = caffe_root + './../XXX.caffemodel'

# net = caffe.Net(model_def,     define the structure of the model
#                 model_weight,  define the trainde weights
#                 caffe.TEST     use test mode 
#                 )


## The end of train the MobileNet



transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
# image_index = 0 # First image in the batch.
# img = transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...]))

import cv2

# plt.figure()
# plt.imshow(img)

# gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
# plt.title('GT: {}'.format(classes[np.where(gtlist)]))
# plt.axis('off');

# plt.show()

def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 1):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        
        for gt, est in zip(gts, ests):
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


for itt in range(1):
    solver.step(50)
    solver.test_nets[0].share_with(solver.net)
    print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 10))



solver.test_nets[0].forward()
test_net = solver.test_nets[0]
print test_net

for image_index in range(10):
    # plt.figure()
    # print image_index, len(test_net.blobs['data'].data)
    img = transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...]))
    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
    print gtlist


    feat = test_net.blobs['score'].data[0]
    print 'feat is: \n'
    # vis_square(feat)
    print test_net.blobs['score'].data[image_index, ...]
    # print test_net.blobs['fc7'].data[image_index, ...]
    estlist = test_net.blobs['score'].data[image_index, ...] > 0
    print ('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
    cv2.imshow(' ',img)
    cv2.waitKey(1000)    
    


