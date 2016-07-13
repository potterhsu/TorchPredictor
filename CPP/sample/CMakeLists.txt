cmake_minimum_required(VERSION 3.2)
project(TorchPredictorSample)

include_directories(../lib)

add_executable(ConvolutionSample ConvolutionSample.cpp)
target_link_libraries(ConvolutionSample TorchPredictor)

add_executable(MaxPoolingSample MaxPoolingSample.cpp)
target_link_libraries(MaxPoolingSample TorchPredictor)

add_executable(ModelForward ModelForward.cpp)
target_link_libraries(ModelForward TorchPredictor)
target_link_libraries(ModelForward ${OpenCV_LIBS})
