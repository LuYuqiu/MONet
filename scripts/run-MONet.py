from __future__ import print_function
import cv2
from scipy import misc
import numpy as np
import tempfile
from math import ceil
import caffe


Frame1 = "/home/user/Luyuqiu/flownet2/data/staticbackground_chair/staticbackground_TEST/120_img1.png"
Frame2 = "/home/user/Luyuqiu/flownet2/data/staticbackground_chair/staticbackground_TEST/120_img2.png"
caffemodel = "/home/user/Luyuqiu/flownet2/models/ObjectNet-SD/MON_SD/_iter_7200.caffemodel"
deployproto = "/home/user/Luyuqiu/flownet2/models/ObjectNet-SD/ObjectNet-sc_deploy.prototxt"
result = "/home/user/Luyuqiu/flownet2/data/staticbackground_chair/test.png"


def load_model(width, height):
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width / divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height / divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    proto = open(deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    caffe.set_logging_disabled()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, caffemodel, caffe.TEST)
    print('Network forward pass using %s.' % caffemodel)

    return net


def prediction():
    
    frame1 = misc.imread(Frame1)
    frame2= misc.imread(Frame2)
    input_data = []

    net = load_model(512, 384)
    blob = predict_gt(frame1, frame2, net)
    gt = np.zeros_like(frame1)
    gt[..., 0] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
    gt[..., 1] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
    gt[..., 2] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(result,gt)
    cv2.imshow("prediction",gt)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def predict_gt(frame1, frame2, net):

    num_blobs = 2
    input_data = [frame1[np.newaxis, :, :, :].transpose(0, 3, 1, 2),
                  frame2[np.newaxis, :, :, :].transpose(0, 3, 1, 2)]  # batch, bgr, h, w

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    # There is some non-deterministic nan-bug in caffe
    i = 1
    while i <= 5:
        i += 1
        net.forward(**input_dict)
        contains_NaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                contains_NaN = True

        if not contains_NaN:
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data)
    return blob


if __name__ == "__main__":
    print("start predict groundtruth")
    prediction()
    
