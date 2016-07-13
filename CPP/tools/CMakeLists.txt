cmake_minimum_required(VERSION 3.2)
project(TorchPredictorTools)

include_directories(../lib)

set(SOURCE_FILES Ascii2BinaryModelConverter.cpp)
add_executable(Ascii2BinaryModelConverter ${SOURCE_FILES})
target_link_libraries(Ascii2BinaryModelConverter TorchPredictor)