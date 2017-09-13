from __future__ import print_function
caffe_root = '/home/toanhoi/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import math
from caffe import layers as L
from caffe.proto import caffe_pb2

def bn_relu_conv_bn_relu(bottom, nout, dropout,split):

    if split == 'train':
        use_global_stats = False
    else:
        use_global_stats=True

    batch_norm1 = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=use_global_stats), in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)])
    scale1 = L.Scale(batch_norm1, bias_term=True, in_place=True,filler=dict(value=1), bias_filler=dict(value=0))
    relu1 = L.ReLU(scale1, in_place=True)
    conv1 = L.Convolution(relu1, kernel_size=[1, 1, 1], pad=[0, 0, 0], stride=[1,1,1],
                          param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                          num_output=nout * 4, axis=1, weight_filler=dict(type='msra'),
                          bias_filler=dict(type='constant'))

    batch_norm2 = L.BatchNorm(conv1, batch_norm_param=dict(use_global_stats=use_global_stats), in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)])
    scale2 = L.Scale(batch_norm2, bias_term=True, in_place=True,filler=dict(value=1), bias_filler=dict(value=0))
    relu2 = L.ReLU(scale2, in_place=True)
    conv2 = L.Convolution(relu2, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                          axis=1, num_output=nout, pad=[1, 1, 1], kernel_size=[3, 3, 3], stride=[1,1,1],
                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    if dropout > 0:
        conv2 = L.Dropout(conv2, dropout_ratio=dropout)
    return conv2


def add_layer(bottom, num_filter, dropout,split):
    conv = bn_relu_conv_bn_relu(bottom, nout=num_filter, dropout=dropout,split=split)
    concate = L.Concat(bottom, conv, axis=1)
    return concate


def transition(bottom, num_filter, split):

    if split == 'train':
        use_global_stats = False
    else:
        use_global_stats=True

    batch_norm1 = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=use_global_stats), in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)])
    scale1 = L.Scale(batch_norm1, bias_term=True, in_place=True,filler=dict(value=1), bias_filler=dict(value=0))
    relu1 = L.ReLU(scale1, in_place=True)
    conv1 = L.Convolution(relu1, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                          axis=1, num_output=num_filter, pad=[0, 0, 0], kernel_size=[1, 1, 1],stride=[1,1,1],
                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    batch_norm2 = L.BatchNorm(conv1, batch_norm_param=dict(use_global_stats=use_global_stats), in_place=False,
                               param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                  dict(lr_mult=0, decay_mult=0)])
    scale2 = L.Scale(batch_norm2, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu2 = L.ReLU(scale2, in_place=True)

    conv_down = L.Convolution(relu2, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                          axis=1, num_output=num_filter, pad=[0, 0, 0], kernel_size=[2, 2, 2], stride=2,
                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))


    #pooling = L.Pooling(conv1, type="Pooling", pool=P.Pooling.MAX, kernel_size=2, stride=2, engine=1)
    return conv_down

# first_output -- #channels before entering the first dense block, set it to be comparable to growth_rate
# growth_rate -- growth rate
# dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(split, batch_size=4, first_output=32, growth_rate=16, dropout=0.2):
    source_train_path = './train_list.txt'
    source_test_path = './test_list.txt'
    patch_size = [64, 64, 64]
    n = caffe.NetSpec()
    num_classes = 4
    reduction= 0.5
    N=[4,4,4,4]
    if split == 'train':
        n.data, n.label = L.HDF5Data(name="data", batch_size=batch_size, source=source_train_path, ntop=2, shuffle=True,
                                     transform_param=dict(crop_size_l=patch_size[0], crop_size_h=patch_size[1],
                                                          crop_size_w=patch_size[2]), include={'phase': caffe.TRAIN})
    elif split == 'val':
        n.data, n.label = L.HDF5Data(name="data", batch_size=batch_size, source=source_test_path, ntop=2, shuffle=True,
                                     transform_param=dict(crop_size_l=patch_size[0], crop_size_h=patch_size[1],
                                                          crop_size_w=patch_size[2]),
                                     include={'phase': caffe.TEST})
    else:
        n.data = L.Input(name="data", ntop=1, input_param={'shape': {'dim': [1, 2, patch_size[0], patch_size[1], patch_size[2]]}})

    nchannels = first_output

    # Fist layers
    n.conv1a = L.Convolution(n.data, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             axis=1, num_output=nchannels, pad=[1,1,1], kernel_size=[3, 3, 3], stride=[1,1,1],
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant',value=-0.1))

    if split == 'train':
        use_global_stats = False
    else:
        use_global_stats=True

    n.bnorm1a = L.BatchNorm(n.conv1a, batch_norm_param=dict(use_global_stats=use_global_stats), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)],  in_place=False)

    n.scale1a = L.Scale(n.bnorm1a, in_place=True, bias_term=True,filler=dict(value=1), bias_filler=dict(value=0))
    n.relu1a = L.ReLU(n.bnorm1a, in_place=True)


    # conv 1b, after BN set bias_term=false
    n.conv1b = L.Convolution(n.relu1a, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                             axis=1, num_output=nchannels, pad=[1, 1, 1], kernel_size=[3, 3, 3], stride=[1,1,1],
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    n.bnorm1b = L.BatchNorm(n.conv1b, batch_norm_param=dict(use_global_stats=use_global_stats), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)], in_place=False)
    n.scale1b = L.Scale(n.bnorm1b, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    n.relu1b = L.ReLU(n.bnorm1b, in_place=True)

    n.conv1c = L.Convolution(n.relu1b, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                             axis=1, num_output=nchannels, pad=[1, 1, 1], kernel_size=[3, 3, 3],stride=[1,1,1],
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    print (nchannels)

    # model = L.Pooling(n.conv1c, type="Pooling", pool=P.Pooling.MAX, kernel_size=2, stride=2, engine=1)

    n.bnorm1c = L.BatchNorm(n.conv1c, batch_norm_param=dict(use_global_stats=use_global_stats), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)], in_place=False)
    n.scale1c = L.Scale(n.bnorm1c, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    n.relu1c = L.ReLU(n.bnorm1c, in_place=True)

    model = L.Convolution(n.relu1c, param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                             axis=1, num_output=nchannels, pad=[0, 0, 0], kernel_size=[2, 2, 2], stride=2,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    n.__setattr__("Conv_down_1", model)

    # ===============Dense block 2=====================
    for i in range(N[0]):
        if (i == 0):
            concat = add_layer(model, growth_rate, dropout,split)
            n.__setattr__("Concat_%d" % (i + 1), concat)
            nchannels += growth_rate
            continue
        concat = add_layer(concat, growth_rate, dropout,split)
        n.__setattr__("Concat_%d" % (i + 1), concat)
        nchannels += growth_rate
    # ===============End dense block 2=================
    print (nchannels)
    # ===============Deconvolution layer 2==============
    model_deconv_x2 = L.Deconvolution(concat, param=[dict(lr_mult=0.1, decay_mult=1)],
                                      convolution_param=dict(kernel_size=[4,4,4], stride=[2,2,2], num_output=num_classes,
                                                             pad=[1, 1, 1], group=num_classes,
                                                             weight_filler=dict(type='bilinear_3D'),
                                                             bias_term=False))
    n.__setattr__("Deconvolution_%d" % (N[0] + 1), model_deconv_x2)
    # ===============End Deconvolution layer 2==============

    # ===============Transition layer 2=================
    model = transition(concat, int(math.floor(nchannels * reduction)), split)
    n.__setattr__("Conv_down_%d" % (N[0] + 1), model)
    nchannels = int(math.floor(nchannels * reduction))
    # ===============End Transition layer2==============

    # ===============Dense block 3=====================
    for i in range(N[1]):
        if (i == 0):
            concat = add_layer(model, growth_rate, dropout, split)
            n.__setattr__("Concat_%d" % (N[1] + i + 2), concat)
            nchannels += growth_rate
            continue
        concat = add_layer(concat, growth_rate, dropout, split)
        n.__setattr__("Concat_%d" % (N[1] + i + 2), concat)
        nchannels += growth_rate
    # ===============End dense block 3=================
    print (nchannels)
    # ===============Deconvolution layer 3==============
    model_deconv_x4 = L.Deconvolution(concat, param=[dict(lr_mult=0.1, decay_mult=1)],
                                      convolution_param=dict(kernel_size=[6,6,6], stride=[4,4,4], num_output=num_classes,
                                                             pad=[1, 1, 1], group=num_classes,
                                                             weight_filler=dict(type='bilinear_3D'),
                                                             bias_term=False))
    n.__setattr__("Deconvolution_%d" % (N[0] + N[1] + 2), model_deconv_x4)
    # ==============Transition layer 3=================
    model = transition(concat, int(math.floor(nchannels * reduction)), split)
    n.__setattr__("Conv_down_%d" % (N[0] + N[1] + 2), model)
    # ===============End Transition layer3==============
    nchannels = int(math.floor(nchannels * reduction))

    # ===============Dense block 4=====================
    for i in range(N[2]):
        if (i == 0):
            concat = add_layer(model, growth_rate, dropout, split)
            n.__setattr__("Concat_%d" % (N[0] + N[1] + i + 3), concat)
            nchannels += growth_rate
            continue
        concat = add_layer(concat, growth_rate, dropout, split)
        n.__setattr__("Concat_%d" % (N[0] + N[1] + i + 3), concat)
        nchannels += growth_rate
    # ===============End dense block 4=================

    # ===============Transition layer 4=================
    print(nchannels)

    # ===============Deconvolution layer 4==============
    model_deconv_x8 = L.Deconvolution(concat, param=[dict(lr_mult=0.1, decay_mult=1)],
                                      convolution_param=dict(kernel_size=[10,10,10], stride=[8,8,8], num_output=num_classes,
                                                             pad=[1, 1, 1], group=num_classes,
                                                             weight_filler=dict(type='bilinear_3D'),
                                                             bias_term=False))
    n.__setattr__("Deconvolution_%d" % (N[0] + N[1] + N[2] + 3), model_deconv_x8)
    # ===============End Deconvolution layer 4==============

    # ===============Transition layer 4=================
    model = transition(concat, int(math.floor(nchannels * reduction)), split)
    n.__setattr__("Conv_down_%d" % (N[0] + N[1] + N[2] + 3), model)
    nchannels = int(math.floor(nchannels * reduction))
    # ===============End Transition layer3==============

    # ===============Dense block 5=====================
    for i in range(N[3]):
        if (i == 0):
            concat = add_layer(model, growth_rate, dropout, split)
            n.__setattr__("Concat_%d" % (N[0] + N[1] + N[2] + N[3] + i + 3), concat)
            nchannels += growth_rate
            continue
        concat = add_layer(concat, growth_rate, dropout, split)
        n.__setattr__("Concat_%d" % (N[0] + N[1] + N[2] + N[3] + i + 3), concat)
        nchannels += growth_rate
    # ===============End dense block 5=================
    print(nchannels)

    # ===============Deconvolution layer 5==============
    model_deconv_x16 = L.Deconvolution(concat, param=[dict(lr_mult=0.1, decay_mult=1)],
                                       convolution_param=dict(kernel_size=[18, 18, 18], stride=[16,16,16],
                                                              num_output=num_classes,
                                                              pad=[1, 1, 1], group=num_classes,
                                                              weight_filler=dict(type='bilinear_3D'),
                                                              bias_term=False))
    n.__setattr__("Deconvolution_%d" % (N[0] + N[1] + N[2] + N[3] + 4), model_deconv_x16)
    model = L.Concat(n.conv1c,model_deconv_x2, model_deconv_x4, model_deconv_x8, model_deconv_x16,
                     axis=1)

    n.bnorm_concat= L.BatchNorm(model, batch_norm_param=dict(use_global_stats=use_global_stats), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)], in_place=False)
    n.scale_concat = L.Scale(n.bnorm_concat, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    n.relu_concat = L.ReLU(n.scale_concat, in_place=True)
    model_conv_concate = L.Convolution(n.relu_concat,
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                       axis=1, num_output=num_classes, pad=[0, 0, 0], kernel_size=[1, 1, 1],
                                       weight_filler=dict(type='msra'))

    if (split == 'train'):
        n.loss = L.SoftmaxWithLoss(model_conv_concate, n.label)

    elif (split == 'val'):
        n.loss = L.SoftmaxWithLoss(model_conv_concate, n.label)
    else:
        n.softmax = L.Softmax(model_conv_concate, ntop=1, in_place=False)
    return n.to_proto()


def make_net():
    with open('train_3d_denseseg.prototxt', 'w') as f:
        print(str(densenet('train', batch_size=4)), file=f)

    with open('test_3d_denseseg.prototxt', 'w') as f:
        print(str(densenet('val', batch_size=4)), file=f)
    with open('deploy_3d_denseseg.prototxt', 'w') as f:
        print(str(densenet('deploy', batch_size=0)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_3d_denseseg.prototxt'

    s.max_iter = 200000
    s.type = 'Adam'
    s.display = 20

    s.base_lr = 0.0002
    #s.power=0.9

    s.momentum = 0.97
    s.weight_decay = 0.0005
    s.average_loss=20
    s.iter_size = 1
    s.lr_policy='step'
    s.stepsize=50000
    s.gamma = 0.1
    s.snapshot_prefix ='./snapshot/3d_denseseg_iseg'
    s.snapshot = 2000
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':
    make_net()
    make_solver()









