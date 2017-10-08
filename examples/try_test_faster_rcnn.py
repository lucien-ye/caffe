import sys
import os

import numpy as np 
import os.path as osp
caffe_root = '../'

sys.path.append(caffe_root + 'python')
sys.path.append("pycaffe") # the tools file is in this folder
sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("/home/dai/caffe/lib/utils")
sys.path.append("/home/dai/caffe/lib/datasets")
sys.path.append("/home/dai/caffe/lib/")
sys.path.append("/home/dai/caffe/tools")





import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import matplotlib.pyplot as plt

from copy import copy
import roi_data_layer.roidb

from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import train_net
import tools
from utils.timer import Timer 
import scipy.io as sio 
import argparse
from fast_rcnn.test import im_detect 
import cv2 



''' for testing stage '''

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, thresh = 0.5):
    ''' Draw detected bounding boxes '''
    inds = np.where(dets[:, -1] >= thresh)[0];
    if len(inds) == 0:
        return
    # im = im[:, :, (2, 1, 0)];
    # fig, ax = plt.subplots(figsize=(12, 12));
    # ax.imshow(im, aspect= 'equal');
    for i in inds:
        bbox = dets[i, :4];
        score = dets[i, -1];
        # print bbox;
        # print score;
        print im.shape;
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1);
        text_name = class_name;  
        # cv2.putText(im, text_name, (bbox[0] + 1/3 * (bbox[2] - bbox[0]), bbox[1] + 1/3 * (bbox[3] - bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255));
        cv2.putText(im, text_name, (int (bbox[0]+(1/3*(bbox[2]-bbox[0]))) , int ( bbox[1] + 1/3 * (bbox[3] - bbox[1])) ), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255));

        cv2.imshow('', im);
        cv2.waitKey(5000);



def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    print "keep is none"
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



# define the Net

cfg.TEST.HAS_RPN = True;

caffe.set_mode_cpu()
# VGG Net
# prototxt = os.path.join('/home/dai/caffe/models/pascal_voc/VGG16/faster_rcnn_alt_opt/', 'faster_rcnn_test.pt');
# caffemodel = os.path.join('/home/dai/caffe/data/faster_rcnn_models/', 'VGG16_faster_rcnn_final.caffemodel');

prototxt = os.path.join('/home/dai/caffe/models/pascal_voc/VGG16/faster_rcnn_end2end/', 'test.prototxt');
caffemodel = os.path.join('/home/dai/caffe/examples/', 'vgg16_faster_rcnn_iter_100.caffemodel');


#MobileNet

#prototxt = os.path.join('/home/dai/caffe/models/pascal_voc/MobileNet/Test/', 'faster_rcnn_test.pt');
# caffemodel = os.path.join('/home/dai/caffe/examples/', 'MobileNet_faster_rcnn_iter_10000.caffemodel');
net = caffe.Net(prototxt, caffemodel, caffe.TEST);

print '\n\nLoaded network {:s}'.format(caffemodel);



# read image and test 
DATA_DIR = '/home/dai/py-faster-rcnn/data/demo/';
image_name = '000456.jpg';
# image_name = '001150.jpg';
# image_name = '001763.jpg';
# image_name = '004545.jpg';
im_file = os.path.join(DATA_DIR, image_name);
im = cv2.imread(im_file);

# cv2.imshow(' ', im);
# cv2.waitKey(1000);

cv2.imshow
timer = Timer();
timer.tic();

scores, boxes = im_detect(net, im);
print "rpn_cls_score is:"
print net.blobs['pool4'].data[0]



timer.toc();
print('Detection took {:.3f}s for ' '{:d} object proposals').format(timer.total_time, boxes.shape[0] );
print('Forward process is done! ');
# print scores;


CONF_THRESH = 0.8;
NMS_THRESH = 0.3;
for cls_ind, cls in enumerate(CLASSES[5:6]):
    cls_ind += 1;
    cls_ind = 6
    cls_boxes = boxes[:, ]
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    # print cls_scores;
    keep = py_cpu_nms(dets, NMS_THRESH);
    if len(keep) != 0:
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)






# ''' for training stage '''

# import pascal_voc_try as PA

# pascal_root = osp.join('/home/dailingzheng/caffe/data/VOC2012/VOCdevkit/VOC2012/')


# classes = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
# if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print("Download pre-trained CaffeNet model ...")

# caffe.set_mode_cpu()


# ## define the cafee model: structure and train and valnet 
# solver = caffe.SGDSolver(osp.join('/home/dai/caffe/models/pascal_voc/VGG16/faster_rcnn_end2end/', 'solver.prototxt'));
# solver.net.copy_from('/home/dai/py-faster-rcnn/data/faster_rcnn_models/' + 'VGG16_faster_rcnn_final.caffemodel');

# imdb_name = "voc_2012_trainval";
# imdb, roidb = train_net.combined_roidb(imdb_name)

# print 'len of roidb is %d' % (len(roidb), );
# # print repr(roidb[0]);

# solver.net.layers[0].set_roidb(roidb)
# # print "net.blobs is:"
# # print solver.net.blobs['data'].data[...]

# solver.step(10);


# # #read the image and  its label, positions
# instance = PA.pascal_voc_try('trainval', '2012')

# import cv2
# for  x in xrange(4,5):
#     print x;
#     print instance.image_path_at(x);
#     image = cv2.imread(instance.image_path_at(x));
#     cv2.imshow( '', image);
#     cv2.waitKey(1000);
#     # print instance._load_pascal_annotation(x)['boxes'];
#     # print instance._load_pascal_annotation(x)['gt_classes'];
#     instance.cache_path = '/home/dai/caffe/data/VOC0712/roidb/';
#     instance.name = 'VOC_12';
#     #print instance.gt_roidb();
#     #print instance.rpn_roidb();











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









