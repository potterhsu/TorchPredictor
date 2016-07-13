//
// Created by Poter Hsu on 2016/3/1.
//

#ifndef BUILD_BINARYMODELPARSER_HPP
#define BUILD_BINARYMODELPARSER_HPP

#include "ModelParser.hpp"
#include <fstream>
#include <assert.h>

class BinaryModelParser : public ModelParser {
public:
    BinaryModelParser(const char *pathToModel_) : ModelParser(pathToModel_) { }

    template <typename T>
    Module<T>* parse() {
        assert(typeid(T) == typeid(double) || typeid(T) == typeid(float));

        Module<T> *model = nullptr;

        fstream fin;
        fin.open(pathToModel, ios::in | ios::binary);
        if (!fin) {
            throw runtime_error("Error: model file not found");
        }

        long type;
        double dvalue;
        while (true) {
            Module<T>* module = nullptr;
            fin.read((char *) &type, sizeof(type));

            if (fin.eof()) break;

            switch (type) {
                case Type::SPATIAL_CONVOLUTION: {
                    // Parse nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH
                    long nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH;
                    fin.read((char *) &nInputPlane, sizeof(nInputPlane));
                    fin.read((char *) &nOutputPlane, sizeof(nOutputPlane));
                    fin.read((char *) &kW, sizeof(kW));
                    fin.read((char *) &kH, sizeof(kH));
                    fin.read((char *) &dW, sizeof(dW));
                    fin.read((char *) &dH, sizeof(dH));
                    fin.read((char *) &padW, sizeof(padW));
                    fin.read((char *) &padH, sizeof(padH));

                    module = new SpatialConvolution<T>(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH);
                    SpatialConvolution<T>* spatialConvolution = (SpatialConvolution<T> *) module;

                    // Parse weight
                    int weightCounter = 0;
                    for (long outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                        for (long inPlane = 0; inPlane < nInputPlane; ++inPlane) {
                            for (long row = 0; row < kH; ++row) {
                                for (long col = 0; col < kW; ++col) {
                                    fin.read((char *) &dvalue, sizeof(dvalue));
                                    spatialConvolution->weight->data[weightCounter] = (T) dvalue;
                                    ++weightCounter;
                                }
                            }
                        }
                    }

                    // Parse bias
                    int biasCounter = 0;
                    for (int outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                        fin.read((char *) &dvalue, sizeof(dvalue));
                        spatialConvolution->bias->data[biasCounter] = (T) dvalue;
                        ++biasCounter;
                    }

                    break;
                }
                case Type::SPATIAL_MAX_POOLING: {
                    long kW, kH, dW, dH, padW, padH;

                    fin.read((char *) &kW, sizeof(kW));
                    fin.read((char *) &kH, sizeof(kH));
                    fin.read((char *) &dW, sizeof(dW));
                    fin.read((char *) &dH, sizeof(dH));
                    fin.read((char *) &padW, sizeof(padW));
                    fin.read((char *) &padH, sizeof(padH));

                    module = new SpatialMaxPooling<T>(kW, kH, dW, dH, padW, padH);
                    break;
                }
                case Type::RELU: {
                    module = new ReLU<T>();
                    break;
                }
                case Type::SOFTMAX: {
                    module = new SoftMax<T>();
                    break;
                }
                case Type::LOG_SOFTMAX: {
                    module = new LogSoftMax<T>();
                    break;
                }
                case Type::VIEW: {
                    long size;

                    // Parse size
                    fin.read((char *) &size, sizeof(size));

                    module = new View<T>(size);
                    break;
                }
                case Type::DROPOUT: {
                    float p;

                    // Parse p
                    fin.read((char *) &dvalue, sizeof(dvalue));
                    p = (float) dvalue;

                    module = new Dropout<T>(p);
                    break;
                }
                case Type::THRESHOLD: {
                    T threshold;

                    // Parse threshold
                    fin.read((char *) &dvalue, sizeof(dvalue));
                    threshold = (T) dvalue;

                    module = new Threshold<T>(threshold);
                    break;
                }
                case Type::LINEAR: {
                    long inputSize, outputSize;

                    // Parse inputSize, outputSize
                    fin.read((char *) &inputSize, sizeof(inputSize));
                    fin.read((char *) &outputSize, sizeof(outputSize));

                    module = new Linear<T>(inputSize, outputSize);
                    Linear<T>* linear = (Linear<T> *) module;

                    // Parse weight
                    int weightCounter = 0;
                    for (long row = 0; row < linear->outputSize; ++row) {
                        for (long col = 0; col < linear->inputSize; ++col) {
                            fin.read((char *) &dvalue, sizeof(dvalue));
                            linear->weight->data[weightCounter++] = (T) dvalue;
                        }
                    }

                    // Parse bias
                    int biasCounter = 0;
                    for (long i = 0; i < linear->outputSize; i++) {
                        fin.read((char *) &dvalue, sizeof(dvalue));
                        linear->bias->data[biasCounter++] = (T) dvalue;
                    }

                    break;
                }
                default:
                    throw "Parse error: Unknown module type";
            }

            module->next = nullptr;
            if (model == nullptr)   model = module;
            else {
                Module<T>* current;
                for (current = model; current->next != nullptr; current = current->next)
                    ;
                current->next = module;
            }
        }   // end of while()

        fin.close();

        return model;
    }
};


#endif //BUILD_BINARYMODELPARSER_HPP
