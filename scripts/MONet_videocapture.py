from __future__ import print_function
import cv2
import numpy as np
import tempfile
from math import ceil
import caffe


VIDEO = "/home/user/Luyuqiu/test/boats/test.avi"
caffemodel = "/home/user/Luyuqiu/flownet2/models/ObjectNet-SD/MON_SD/_iter_7200.caffemodel"
deployproto = "/home/user/Luyuqiu/flownet2/models/ObjectNet-SD/ObjectNet-s_deploy.prototxt"
WRITER = "/home/user/Luyuqiu/test/boats/s_prediction.avi"

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


def video2gt():
    cap = cv2.VideoCapture(VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 8.0
    writer = cv2.VideoWriter(WRITER, fourcc, fps, (width/2, height))

    _, frame1 = cap.read()
    _, frame2 = cap.read()

    net = load_model(width, height)
    while True:
        if frame2 is None:
            break
        blob = predict_gt(frame1, frame2, net)
        gt = np.zeros_like(frame1)
        gt[..., 0] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
        gt[..., 1] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
        gt[..., 2] = cv2.normalize(blob, None, 0, 255, cv2.NORM_MINMAX)
        hmerge = np.vstack((frame1, gt))
        hmerge = cv2.resize(hmerge, (width/2, height))
        cv2.imshow("frame-gt", hmerge)
        writer.write(hmerge)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        frame1 = frame2
        _, frame2 = cap.read()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def predict_gt(frame0, frame1, net):

    num_blobs = 2
    input_data = [frame0[np.newaxis, :, :, :].transpose(0, 3, 1, 2),
                  frame1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)]  # batch, bgr, h, w

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
    video2gt()
    
