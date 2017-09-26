#encoding=utf8

import lmdb
import caffe
import numpy as np
import sys
import cv2

if len(sys.argv) < 2:
    sys.stdcerr('error')

lmdb_path = sys.argv[1]


os.path.join(lmdb_path, )
env = lmdb.Environment(lmdb_path , map_size=8*3*11540*227*227)


with env.begin(write=True) as txn:
    
    datum = caffe.proto.caffe_pb2.Datum()
    img = cv2
