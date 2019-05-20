import lmdb
import caffe
from caffe.proto import caffe_pb2
import cv2 as cv

env = lmdb.open("/home/user/Luyuqiu/flownet2/data/staticbackground_chair/TEST_lmdb", readonly=True)
txn = env.begin()
cur = txn.cursor()
datum = caffe_pb2.Datum() # caffe 定义的数据类型


for key, value in cur:
    print(type(key), key)
    datum.ParseFromString(value) # 反序列化成 datum 对象

    data = caffe.io.datum_to_array(datum)
    print(data.shape)
    print(datum.channels)
    gt = data[6]
    print(gt.shape)
    cv.imshow("gt",gt)
    cv.waitKey(0)

cv2.destroyAllWindows()
env.close()
