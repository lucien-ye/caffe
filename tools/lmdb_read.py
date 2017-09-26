#encoding=utf8

import lmdb
import numpy as np
import caffe
import sys
from matplotlib import pyplot as plt


if len(sys.argv) < 2:
    sys.stdcerr("lmdb_read.py lmdb/file/path")
    sys.exit(0)


lmdb_path = sys.argv[1]
print 'reading ' + lmdb_path + '....'

env = lmdb.open(lmdb_path, readonly=True)

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print 'key: ', key
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        flat_img = np.fromstring(datum.data, dtype=np.uint8)
        img = flat_img.reshape(datum.channels, datum.height, datum.width)
        y = datum.label
        fig = plt.figure()
        plt.imshow(img, cmap='gray')




