# README #

### Features ###

* A C++ library for loading and predicting your model trained by Torch7
* Only converting model needs Torch7, it's independence from Torch7 when running model
* Reasonably fast, without GPU, just a little slower than Torch7
* Currently CPU only

### Dependencies ###

* OpenBLAS
* OpenMP
* Torch7 (needs for converting model)

### Setup ###

1. $ mkdir build; cd build
2. $ cmake -D CMAKE_CXX_COMPILER=/path/to/clang-omp++ ../
3. $ make
4. $ make install

### Usage ###

1. Get tpa model: edit and run 2_convert-t7-to-tpa.sh under Torch folder
2. Get tpb model: run Ascii2BinaryModelConverter under tools folder
3. See ModelForward under sample folder to load binary model (or you can just load ascii model)

### Supported Networks ###

##### Layers #####

* SpatialConvolution
* SpatialMaxPooling
* Linear
* View
* Threshold
* Dropout 


##### Activation Functions #####

* ReLU
* SoftMax
* LogSoftMax


### Demo ###

* See [TorchPredictorDemo](https://github.com/potterhsu/TorchPredictorDemo)
