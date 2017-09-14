# 3D_DenseSeg: 3D Densely Convolutional Networks for Volumetric Segmentation
By Toan Duc Bui, Jitae Shin, Taesup Moon

This is the implementation of our method in the MICCAI Grand Challenge on 6-month infant brain MRI segmentation-in conjunction with MICCAI 2017.

Link: https://arxiv.org/abs/1709.03199

### Introduction
6-month infant brain MRI segmentation aims to segment the brain into: White matter, Gray matter, and Cerebrospinal fluid. It is a difficult task due to larger overlapping between tissues, low contrast intensity. We treat the problem by using very deep 3D convolution neural network. Our result achieved the top performance in 6 performance metrics. 

### Dice Coefficient (DC) for 9th subject
|                   | CSF       | GM             | WM   | Average 
|-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|
|3D-DenseSeg  | 94.74% | 91.61% |91.30% | 92.55% 

### Citation

### Requirements: 
- 3D-CAFFE (as below), python 2.7, Ubuntu 14.04, CUDNN 5.1, CUDA 8.0
- TiTan X Pascal 12GB

### Installation
- Step 1: Download the source code
```
git clone https://github.com/tbuikr/3D_DenseSeg.git
cd 3D_DenseSeg
```
- Step 2: Download dataset at `http://iseg2017.web.unc.edu/download/` and change the path of the dataset `data_path` and saved path `target_path` in file `prepare_hdf5_cutedge.py`
```
data_path = '/path/to/your/dataset/'
target_path = '/path/to/your/save/hdf5 folder/'
```

- Step 3: Generate hdf5 dataset

```
python prepare_hdf5_cutedge.py
```

- Step 4: Run training

```
./run_train.sh
```

- Step 5: Generate score map and segmentation image. You have to change the path in the file `seg_deploy.py` as 
```data_path = '/path/to/your/dataset/'
caffe_root = '/path/to/your/caffe/build/tools/caffe/'# (i.e '/home/toanhoi/caffe/build/tools/caffe/')
```

And run
```
python seg_deploy.py
```

### 3D CAFFE
For CAFFE, we use 3D UNet CAFFE with minor modification. Hence, you first download the 3D UNet CAFFE at

`https://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html`

And run the installation as the README file. Then we change the HDF5DataLayer that allows to randomly crop patch based on the code at `https://github.com/yulequan/3D-Caffe`
You can download the code by
```
git clone https://github.com/yulequan/3D-Caffe/
cd 3D-Caffe
git checkout 3D-Caffe
cd ../
```

After downloading both source codes, we have two folder code `3D-Caffe` and `caffe` (for 3D UNet CAFFE). We have to copy the hdf5 data files from `3D-Caffe` to `caffe` by the commands

```
cp ./3D-Caffe/src/caffe/layers/hdf5_data_layer.cpp ./caffe/src/caffe/layers/
cp ./3D-Caffe/src/caffe/layers/hdf5_data_layer.cu ./caffe/src/caffe/layers/
cp ./3D-Caffe/include/caffe/layers/hdf5_data_layer.hpp ./caffe/include/caffe/layers/hdf5_data_layer.hpp
```

Then add these lines in the field `message TransformationParameter` of the file  `caffe.proto` in the `./caffe/src/caffe/proto`
 (3D UNet CAFFE)
```
optional uint32 crop_size_w = 8 [default = 0];
optional uint32 crop_size_h = 9 [default = 0];
optional uint32 crop_size_l = 10 [default = 0];
```

Add following code in the `./caffe/include/caffe/filler.hpp`

```
/**
  3D bilinear filler 
*/
template <typename Dtype>
class BilinearFiller_3D : public Filler<Dtype> {
 public:
  explicit BilinearFiller_3D(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 5) << "Blob must be 5 dim.";
    CHECK_EQ(blob->shape(-1), blob->shape(-2)) << "Filter must be square";
    CHECK_EQ(blob->shape(-2), blob->shape(-3)) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();

    int f = ceil(blob->shape(-1) / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->shape(-1);
      float y = (i / blob->shape(-1)) % blob->shape(-2);
      float z = (i/(blob->shape(-1)*blob->shape(-2))) % blob->shape(-3);
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c)) * (1-fabs(z / f - c));
    }
    

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};
```

and in the `GetFiller(const FillerParameter& param)` function (same file)

```
else if (type == "bilinear_3D"){
    return new BilinearFiller_3D<Dtype>(param);
  }
 ```

Final, you recompile 3D UNet CAFFE (uncomment `USE_CUDNN := 1`) and can you my prototxt. Please cite these papers when you use the CAFFE code

###
### Note
- If you want to generate network prototxt, you have to change the path of `caffe_root`
```
caffe_root = '/path/to/your/caffe/build/tools/caffe/'# (i.e '/home/toanhoi/caffe/build/tools/caffe/')
```
And run
```
python make_3D_DenseSeg.py
```
- If you have the error `AttributeError: 'LayerParameter' object has no attribute 'shuffle'` when run  `python make_3D_DenseSeg.py`, then you can fix it by replacing the line 35 in the `net_spec.py`:
```
  #param_names = [s for s in dir(layer) if s.endswith('_param')]
  param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
  ```
- Plot training loss during training

```
python plot_trainingloss.py ./log/3D_DenseSeg.log 
```

