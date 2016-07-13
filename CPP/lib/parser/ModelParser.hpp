//
// Created by Poter Hsu on 2016/3/3.
//

#ifndef BUILD_MODELPARSER_HPP
#define BUILD_MODELPARSER_HPP

#include "../nn/Module.hpp"
#include "../nn/SpatialConvolution.hpp"
#include "../nn/SpatialMaxPooling.hpp"
#include "../nn/ReLU.hpp"
#include "../nn/SoftMax.hpp"
#include "../nn/LogSoftMax.hpp"
#include "../nn/View.hpp"
#include "../nn/Dropout.hpp"
#include "../nn/Threshold.hpp"
#include "../nn/Linear.hpp"

class ModelParser {
protected:
    const char* pathToModel;

public:

    ModelParser(const char* pathToModel_);

    virtual ~ModelParser();
};


#endif //BUILD_MODELPARSER_HPP
