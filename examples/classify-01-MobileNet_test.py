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

# main netspec wrapper

workdir = './MobileNet_pascal_classify'


#define the Net 
caffe.set_mode_cpu()
model_weight = workdir + '/snapshot_iter_5000.caffemodel'
model_def = workdir + '/MoblieNet_deploy.prototxt'

net = caffe.Net(model_def,     #define the structure of the model
                model_weight,  #define the trainde weights
                caffe.TEST     #use test mode 
                )


## The end of train the MobileNet


#transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

#image = transformer.

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

net.blobs['data'].reshape(1,
                          3,
                          224, 224)


image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
import cv2

net.blobs['data'].data[...] = transformed_image
output = net.forward()

print net.blobs['score'].data[0]
print "output is done!! "

estlist = net.blobs['score'].data[0] > 0
print estlist
print ('predict is {}' .format(*classes[np.where(estlist)]) )

cv2.imshow(' ',image)
cv2.waitKey(1000)    








# def hamming_distance(gt, est):
#     return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

# def check_accuracy(net, num_batches, batch_size = 1):
#     acc = 0.0
#     for t in range(num_batches):
#         net.forward()
#         gts = net.blobs['label'].data
#         ests = net.blobs['score'].data > 0
        
#         for gt, est in zip(gts, ests):
#             acc += hamming_distance(gt, est)
#     return acc / (num_batches * batch_size)


# for itt in range(1):
#     #solver.step(50)
#     solver.test_nets[0].share_with(solver.net)
#     print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 10))



# solver.test_nets[0].forward()
# test_net = solver.test_nets[0]
# print test_net

# for image_index in range(10):
#     # plt.figure()
#     # print image_index, len(test_net.blobs['data'].data)
#     img = transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...]))
#     gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
#     print gtlist


#     feat = test_net.blobs['score'].data[0]
#     print 'feat is: \n'
#     # vis_square(feat)
#     print test_net.blobs['score'].data[image_index, ...]
#     # print test_net.blobs['fc7'].data[image_index, ...]
#     estlist = test_net.blobs['score'].data[image_index, ...] > 0
#     print ('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
#     cv2.imshow(' ',img)
#     cv2.waitKey(1000)    
    


