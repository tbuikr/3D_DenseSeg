from medpy.io import load
import numpy as np
import os
import h5py

#Path to your dataset (img, hdr files)
data_path = '/media/toanhoi/Study/databaseSeg/ISeg/iSeg-2017-Training'
#Saved path
target_path = '/media/toanhoi/Study/databaseSeg/ISeg/hdf5_iseg_data'
#Reference https://github.com/zhengyang-wang/Unet_3D/tree/master/preprocessing
def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def convert_label(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:, :, i]=label_slice
    return label_processed

def build_h5_dataset(data_path, target_path):
    '''
    Build HDF5 Image Dataset.
    '''
    for i in range(10):
        #Skip subject 9 for validation
        if (i==8):
            continue
        subject_name = 'subject-%d-' % (i + 1)
        f_T1 = os.path.join(data_path, subject_name + 'T1.hdr')
        img_T1, header_T1 = load(f_T1)
        f_T2 = os.path.join(data_path, subject_name + 'T2.hdr')
        img_T2, header_T2 = load(f_T2)
        f_l = os.path.join(data_path, subject_name + 'label.hdr')
        labels, header_label = load(f_l)

        inputs_T1 = img_T1.astype(np.float32)
        inputs_T2 = img_T2.astype(np.float32)
        labels = labels.astype(np.uint8)
        labels=convert_label(labels)
        mask=labels>0
        # Normalization
        inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
        inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()

        # Cut edge
        margin = 64/2   # training_patch_size / 2
        mask = mask.astype(np.uint8)
        min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(mask, margin)
        inputs_tmp_T1 = inputs_T1_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
        inputs_tmp_T2 = inputs_T2_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

        labels_tmp = labels[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

        inputs_tmp_T1 = inputs_tmp_T1[:, :, :, None]
        inputs_tmp_T2 = inputs_tmp_T2[:, :, :, None]
        labels_tmp = labels_tmp[:, :, :, None]

        inputs = np.concatenate((inputs_tmp_T1, inputs_tmp_T2), axis=3)

        print (inputs.shape, labels_tmp.shape)

        inputs_caffe = inputs[None, :, :, :, :]
        labels_caffe = labels_tmp[None, :, :, :, :]
        inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
        labels_caffe = labels_caffe.transpose(0, 4, 3, 1, 2)

        with h5py.File(os.path.join(target_path, 'train_iseg_norm_cutedge_weight_%s.h5' % (i+1)), 'w') as f:
            f['data'] = inputs_caffe  # for caffe num channel x d x h x w
            f['label'] = labels_caffe

        with open('./train_list.txt', 'a') as f:
            f.write(os.path.join(target_path, 'train_iseg_norm_cutedge_weight_%s.h5\n' % (i+1)))

if __name__ == '__main__':
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists("./train_list.txt"):
        os.remove("./train_list.txt")
    build_h5_dataset(data_path, target_path)
