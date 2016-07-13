cmake_minimum_required(VERSION 3.2)
project(TorchPredictorTest)

include_directories(../tools)
include_directories(../lib)

set(SOURCE_FILES TestAll.cpp AsciiModelParserTest.cpp AsciiModelParserTest.h SpatialConvolutionTest.cpp SpatialConvolutionTest.h SpatialMaxPoolingTest.cpp SpatialMaxPoolingTest.h ReLUTest.cpp ReLUTest.h SoftMaxTest.cpp SoftMaxTest.h LogSoftMaxTest.cpp LogSoftMaxTest.h ViewTest.cpp ViewTest.h DropoutTest.cpp DropoutTest.h ThresholdTest.cpp ThresholdTest.h LinearTest.cpp LinearTest.h BinaryModelParserTest.cpp BinaryModelParserTest.h)
add_executable(TorchPredictorTest ${SOURCE_FILES})
target_link_libraries(TorchPredictorTest gtest)
target_link_libraries(TorchPredictorTest TorchPredictor)
target_link_libraries(TorchPredictorTest ${OpenCV_LIBS})