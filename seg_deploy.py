import csv
caffe_root = '/home/toanhoi/caffe/build/tools/caffe/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "1"
import caffe
import argparse
import numpy as np
from medpy.io import load, save

#0:     Background (everything outside the brain)
#10:   Cerebrospinal fluid (CSF)
#150: Gray matter (GM)
#250: White matter (WM)
def convert_label_submit(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1] = 10
        label_slice[label_slice == 2] = 150
        label_slice[label_slice == 3] = 250
        label_processed[:, :, i]=label_slice
    return label_processed

def convert_label(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:, :, i]=label_slice
    return label_processed
#Reference https://github.com/ginobilinie/infantSeg
def dice(im1, im2,tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
    parser.add_argument("-start")
    parser.add_argument("-end")

    if (os.environ.get('CAFFE_CPU_MODE')):
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    data_path = '/media/toanhoi/Study/databaseSeg/ISeg/iSeg-2017-Training'

    subject_id=9
    subject_name = 'subject-%d-' % subject_id

    f_T1 = os.path.join(data_path, subject_name + 'T1.hdr')
    inputs_T1, header_T1 = load(f_T1)
    inputs_T1 = inputs_T1.astype(np.float32)

    f_T2 = os.path.join(data_path, subject_name + 'T2.hdr')
    inputs_T2, header_T2 = load(f_T2)
    inputs_T2 = inputs_T2.astype(np.float32)

    f_l = os.path.join(data_path, subject_name + 'label.hdr')
    labels, header_label = load(f_l)
    labels = labels.astype(np.uint8)
    labels=convert_label(labels)

    mask = inputs_T1 > 0
    mask = mask.astype(np.bool)
    # ======================normalize to 0 mean and 1 variance====
    # Normalization
    inputs_T1_norm =(inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
    inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()

    inputs_T1_norm = inputs_T1_norm[:, :, :, None]
    inputs_T2_norm = inputs_T2_norm[:, :, :, None]

    inputs = np.concatenate((inputs_T1_norm, inputs_T2_norm), axis=3)
    inputs = inputs[None, :, :, :, :]
    inputs = inputs.transpose(0, 4, 3, 1, 2)
    num_class=4
    num_paches=0

    model_def='./deploy_3d_denseseg.prototxt'
    model_weights = "./snapshot/3d_denseseg_iseg_iter_168000.caffemodel"
    net = caffe.Net(model_def, model_weights,caffe.TEST)
    patch_input = [64, 64, 64]
    xstep = 16
    ystep = 8#16
    zstep = 16#16
    deep_slices    = np.arange(patch_input[0] // 2, inputs.shape[2] - patch_input[0] // 2 + xstep, xstep)
    height_slices  = np.arange(patch_input[1] // 2, inputs.shape[3] - patch_input[1] // 2 + ystep, ystep)
    width_slices   = np.arange(patch_input[2] // 2, inputs.shape[4] - patch_input[2] // 2 + zstep, zstep)
    output = np.zeros((num_class,) + inputs.shape[2:])
    count_used=np.zeros((inputs.shape[2],inputs.shape[3],inputs.shape[4]))+1e-5

    total_patch=len(deep_slices)*len(height_slices)*len(width_slices)
    for i in range(len(deep_slices)):
        for j in range(len(height_slices)):
            for k in range(len(width_slices)):
                num_paches=num_paches+1
                deep   = deep_slices[i]
                height = height_slices[j]
                width  = width_slices[k]
                raw_patches= inputs[:,:,deep - patch_input[0] // 2:deep + patch_input[0] // 2,
                                         height - patch_input[1] // 2:height + patch_input[1] // 2,
                                         width - patch_input[2] // 2:width + patch_input[2] // 2]
                print "Processed: ",num_paches ,"/", total_patch
                net.blobs['data'].data[...] = raw_patches
                net.forward()

                #Major voting https://github.com/ginobilinie/infantSeg
                temp_predic=net.blobs['softmax'].data[0].argmax(axis=0)
                for labelInd in range(4):  # note, start from 0
                    currLabelMat = np.where(temp_predic == labelInd, 1, 0)  # true, vote for 1, otherwise 0
                    output[labelInd, deep - patch_input[0] // 2:deep + patch_input[0] // 2,
                        height - patch_input[1] // 2:height + patch_input[1] // 2,
                        width - patch_input[2] // 2:width + patch_input[2] // 2] += currLabelMat
                #Average
                # output[slice(None),deep - patch_input[0] // 2:deep + patch_input[0] // 2,
                #         height - patch_input[1] // 2:height + patch_input[1] // 2,
                #         width - patch_input[2] // 2:width + patch_input[2] // 2]+=net.blobs['softmax'].data[0]

                count_used[deep - patch_input[0] // 2:deep + patch_input[0] // 2,
                        height - patch_input[1] // 2:height + patch_input[1] // 2,
                        width - patch_input[2] // 2:width + patch_input[2] // 2]+=1

    output=output/count_used
    y = np.argmax(output, axis=0)
    out_label=y.transpose(1,2,0)
    dsc_0 = dice(out_label , labels, 0)
    dsc_1 = dice(out_label , labels, 1)
    dsc_2 = dice(out_label , labels, 2)
    dsc_3 = dice(out_label , labels, 3)
    dsc = np.mean([dsc_1, dsc_2, dsc_3])  # ignore Background
    print dsc_1, dsc_2, dsc_3, dsc
    with open('result_3d_dense_seg.csv', 'a+') as csvfile:
        datacsv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        datacsv.writerow([dsc_1, dsc_2, dsc_3, dsc])

    out_label=out_label.astype(np.uint8)
    out_label = convert_label_submit(out_label)
    save(out_label, '{}/{}'.format("./", "3d_dense_seg_result.hdr"), header_T1)