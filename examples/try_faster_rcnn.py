import sys
import os

import numpy as np 
import os.path as osp
caffe_root = '../'



sys.path.append(caffe_root + 'python')
sys.path.append("pycaffe") # the tools file is in this folder
sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("../lib/utils")
sys.path.append("../lib/datasets")
sys.path.append("../lib")
sys.path.append("../tools")
sys.path.append("../py_tools")

import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import matplotlib.pyplot as plt

from copy import copy
import roi_data_layer.roidb

from fast_rcnn.train import get_training_roidb, train_net, filter_roidb

from fast_rcnn.train import train_net  as trainnet

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb



# import pascal_voc_try as PA

import tools
import train_net

pascal_root = osp.join('/home/dailingzheng/caffe/data/VOCkls2012/VOCdevkit/VOC2012/')


caffe.set_mode_cpu()

cfg_file =  '../experiments/cfgs/faster_rcnn_end2end.yml'; 
cfg_from_file(cfg_file);


## define the cafee model: structure and train and valnet 
# solver = caffe.SGDSolver(osp.join('/home/dai/caffe/models/pascal_voc/VGG16/faster_rcnn_end2end/', 'solver.prototxt'));

# based on MobileNet
#solver = caffe.SGDSolver(osp.join('/home/dai/caffe/models/pascal_voc/MobileNet/faster_rcnn_end2end/', 'solver.prototxt'));
#solver.net.copy_from('/home/dai/caffe/models/mobilenet//' + 'mobilenet.caffemodel');
# pretrained_model = '/home/dai/caffe/data/faster_rcnn_models/' + 'VGG16_faster_rcnn_final.caffemodel';



# based on MobileNet
#solver = caffe.SGDSolver(osp.join('/home/dai/caffe/models/pascal_voc/MobileNet/faster_rcnn_end2end/', 'solver.prototxt'));
# solver.net.copy_from('/home/dai/caffe/models/mobilenet//' + 'mobilenet.caffemodel');

# solver.net.copy_from('/home/dai/py-faster-rcnn/data/faster_rcnn_models/' + 'VGG16_faster_rcnn_final.caffemodel');

imdb_name = "voc_2012_trainval";
imdb, roidb = train_net.combined_roidb(imdb_name)

# print 'len of roidb is %d' % (len(roidb), );
# roidb = filter_roidb(roidb);
# print repr(roidb[0]);

# solver.net.layers[0].set_roidb(roidb)
# solver.step(11000);





# train_net(solver, roidb, output_dir,
#           pretrained_model=pretrained_model,
#           max_iters=max_iters);

solver = '../models/pascal_voc/VGG16/faster_rcnn_end2end/' + 'solver.prototxt';
pretrained_model = '../data/faster_rcnn_models/' + 'VGG16_faster_rcnn_final.caffemodel';

# print "net.blobs is:"
# print solver.net.blobs['data'].data[...]
max_iters = 1000;
output_dir = '../';

trainnet(solver, roidb, output_dir,
          pretrained_model=pretrained_model,
          max_iters=max_iters);




# #read the image and  its label, positions
instance = PA.pascal_voc_try('trainval', '2012')

import cv2
for  x in xrange(4,5):
    print x;
    print instance.image_path_at(x);
    image = cv2.imread(instance.image_path_at(x));
    cv2.imshow( '', image);
    cv2.waitKey(1000);
    # print instance._load_pascal_annotation(x)['boxes'];
    # print instance._load_pascal_annotation(x)['gt_classes'];
    instance.cache_path = '/home/dai/caffe/data/VOC0712/roidb/';
    instance.name = 'VOC_12';
    #print instance.gt_roidb();
    #print instance.rpn_roidb();




# # main netspec wrapper


# workdir = './MobileNet_pascal_classify'
# if not os.path.isdir(workdir):
#     os.makedirs(workdir)

# solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), 
#                                          testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
# solverprototxt.sp['display'] = "10"
# solverprototxt.sp['base_lr'] = "0.0001"
# solverprototxt.write(osp.join(workdir, 'solver.prototxt'))



# solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
# solver.net.copy_from(caffe_root + 'models/mobilenet/mobilenet.caffemodel')

# #define the Net 
# # caffe.set_mode_cpu
# # model_def = caffe_root + '../../deploy.prototxt'
# # model_weight = caffe_root + './../XXX.caffemodel'

# # net = caffe.Net(model_def,     define the structure of the model
# #                 model_weight,  define the trainde weights
# #                 caffe.TEST     use test mode 
# #                 )

# ## The end of train the MobileNet

# transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
# # image_index = 0 # First image in the batch.
# # img = transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...]))

# import cv2

# # plt.figure()
# # plt.imshow(img)

# # gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
# # plt.title('GT: {}'.format(classes[np.where(gtlist)]))
# # plt.axis('off');

# # plt.show()

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
#     solver.step(50)
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
    






# #define the Net 
# caffe.set_mode_cpu()
# model_weight = workdir + '/snapshot_iter_5000.caffemodel'
# model_def = workdir + '/MoblieNet_deploy.prototxt'

# net = caffe.Net(model_def,     #define the structure of the model
#                 model_weight,  #define the trainde weights
#                 caffe.TEST     #use test mode 
#                 )


# ## The end of train the MobileNet


# #transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

# #image = transformer.

# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print 'mean-subtracted values:', zip('BGR', mu)

# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', mu)
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2, 1, 0))

# net.blobs['data'].reshape(1,
#                           3,
#                           224, 224)


# image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# transformed_image = transformer.preprocess('data', image)
# import cv2

# net.blobs['data'].data[...] = transformed_image
# output = net.forward()

# print net.blobs['score'].data[0]
# print "output is done!! "

# estlist = net.blobs['score'].data[0] > 0
# print estlist
# print ('predict is {}' .format(*classes[np.where(estlist)]) )

# cv2.imshow(' ',image)
# cv2.waitKey(1000)    









