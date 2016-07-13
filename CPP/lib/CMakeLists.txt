cmake_minimum_required(VERSION 3.2)
project(TorchPredictor)

set(SOURCE_FILES parser/ModelParser.cpp)
set(HEADER_FILES parser/ModelParser.hpp parser/AsciiModelParser.hpp parser/BinaryModelParser.hpp tensor/Tensor.hpp nn/Type.hpp nn/Module.hpp nn/SpatialConvolution.hpp nn/SpatialMaxPooling.hpp nn/ReLU nn/SoftMax.hpp nn/LogSoftMax.hpp nn/View.hpp nn/Dropout.hpp nn/Threshold.hpp nn/Linear.hpp)
add_library(TorchPredictor ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(TorchPredictor
        PROPERTIES
        PUBLIC_HEADER ${HEADER_FILES})
target_link_libraries(TorchPredictor openblas)

install(TARGETS TorchPredictor
        ARCHIVE DESTINATION lib)

install(DIRECTORY ./ DESTINATION include/TorchPredictor
        FILES_MATCHING PATTERN "*.hpp")

