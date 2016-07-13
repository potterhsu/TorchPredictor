//
// Created by Poter Hsu on 2015/12/11.
//

#ifndef BUILD_ASCIIMODELPARSER_H
#define BUILD_ASCIIMODELPARSER_H

#include "ModelParser.hpp"
#include <fstream>
#include <assert.h>

class AsciiModelParser : public ModelParser {

public:
    AsciiModelParser(const char *pathToModel_) : ModelParser(pathToModel_) { }

    template <typename T>
    Module<T>* parse() {
        assert(typeid(T) == typeid(double) || typeid(T) == typeid(float));

        Module<T> *model = nullptr;

        fstream fin;
        fin.open(pathToModel, ios::in);
        if (!fin) {
            throw runtime_error("Error: model file not found");
        }

        char buffer[BUFSIZ];
        while (fin.getline(buffer, sizeof(buffer), '\n')) {
            Module<T>* module = nullptr;

            if (strstr(buffer, "nn.Sequential") != nullptr) {
                // Do nothing
            }
            else if (strstr(buffer, "nn.SpatialConvolution") != nullptr) {
                // Parse nInputPlane
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "nInputPlane") == nullptr)
                    throw "Parse error: keyword nInputPlane not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long nInputPlane = (long) atoi(buffer);

                // Parse nOutputPlane
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "nOutputPlane") == nullptr)
                    throw "Parse error: keyword nOutputPlane not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long nOutputPlane = (long) atoi(buffer);

                // Parse kW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "kW") == nullptr)
                    throw "Parse error: keyword kW not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long kW = (long) atoi(buffer);

                // Parse kH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "kH") == nullptr)
                    throw "Parse error: keyword kH not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long kH = (long) atoi(buffer);

                // Parse dW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "dW") == nullptr)
                    throw "Parse error: keyword dW not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long dW = (long) atoi(buffer);

                // Parse dH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "dH") == nullptr)
                    throw "Parse error: keyword dH not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long dH = (long) atoi(buffer);

                // Parse padW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "padW") == nullptr)
                    throw "Parse error: keyword padW not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long padW = (long) atoi(buffer);

                // Parse padH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "padH") == nullptr)
                    throw "Parse error: keyword padH not found in SpatialConvolution";
                fin.getline(buffer, sizeof(buffer), '\n');
                long padH = (long) atoi(buffer);

                module = new SpatialConvolution<T>(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH);
                SpatialConvolution<T>* spatialConvolution = (SpatialConvolution<T> *) module;

                // Parse weight
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "weight") == nullptr)
                    throw "Parse error: keyword weight not found in SpatialConvolution";
                int weightCounter = 0;
                for (long outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                    for (long inPlane = 0; inPlane < nInputPlane; ++inPlane) {
                        for (long row = 0; row < kH; ++row) {
                            for (long col = 0; col < kW; ++col) {
                                fin.getline(buffer, sizeof(buffer), '\n');
                                spatialConvolution->weight->data[weightCounter] = (T) atof(buffer);
                                ++weightCounter;
                            }
                        }
                    }
                }

                // Parse bias
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "bias") == nullptr)
                    throw "Parse error: keyword bias not found in SpatialConvolution";
                int biasCounter = 0;
                for (int outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                    fin.getline(buffer, sizeof(buffer), '\n');
                    spatialConvolution->bias->data[biasCounter] = (T) atof(buffer);
                    ++biasCounter;
                }
            }
            else if (strstr(buffer, "nn.SpatialMaxPooling") != nullptr) {
                // Parse kW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "kW") == nullptr)
                    throw "Parse error: keyword kW not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long kW = (long) atoi(buffer);

                // Parse kH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "kH") == nullptr)
                    throw "Parse error: keyword kH not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long kH = (long) atoi(buffer);

                // Parse dW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "dW") == nullptr)
                    throw "Parse error: keyword dW not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long dW = (long) atoi(buffer);

                // Parse dH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "dH") == nullptr)
                    throw "Parse error: keyword dH not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long dH = (long) atoi(buffer);

                // Parse padW
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "padW") == nullptr)
                    throw "Parse error: keyword padW not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long padW = (long) atoi(buffer);

                // Parse padH
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "padH") == nullptr)
                    throw "Parse error: keyword padH not found in SpatialMaxPooling";
                fin.getline(buffer, sizeof(buffer), '\n');
                long padH = (long) atoi(buffer);

                module = new SpatialMaxPooling<T>(kW, kH, dW, dH, padW, padH);
            }
            else if (strstr(buffer, "nn.ReLU") != nullptr) {
                module = new ReLU<T>();
            }
            else if (strstr(buffer, "nn.SoftMax") != nullptr) {
                module = new SoftMax<T>();
            }
            else if (strstr(buffer, "nn.LogSoftMax") != nullptr) {
                module = new LogSoftMax<T>();
            }
            else if (strstr(buffer, "nn.View") != nullptr) {
                // Parse size
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "size") == nullptr)
                    throw "Parse error: keyword size not found in View";
                fin.getline(buffer, sizeof(buffer), '\n');
                long size = (long) atoi(buffer);

                module = new View<T>(size);
            }
            else if (strstr(buffer, "nn.Dropout") != nullptr) {
                // Parse p
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "p") == nullptr)
                    throw "Parse error: keyword p not found in Dropout";
                fin.getline(buffer, sizeof(buffer), '\n');
                float p = (float) atof(buffer);

                module = new Dropout<T>(p);
            }
            else if (strstr(buffer, "nn.Threshold") != nullptr) {
                // Parse threshold
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "threshold") == nullptr)
                    throw "Parse error: keyword threshold not found in Threshold";
                fin.getline(buffer, sizeof(buffer), '\n');
                T th = (T) atof(buffer);

                module = new Threshold<T>(th);
            }
            else if (strstr(buffer, "nn.Linear") != nullptr) {
                // Parse inputSize
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "inputSize") == nullptr)
                    throw "Parse error: keyword inputSize not found in Linear";
                fin.getline(buffer, sizeof(buffer), '\n');
                long inputSize = (long) atoi(buffer);

                // Parse outputSize
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "outputSize") == nullptr)
                    throw "Parse error: keyword outputSize not found in Linear";
                fin.getline(buffer, sizeof(buffer), '\n');
                long outputSize = (long) atoi(buffer);

                module = new Linear<T>(inputSize, outputSize);
                Linear<T>* linear = (Linear<T> *) module;

                // Parse weight
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "weight") == nullptr)
                    throw "Parse error: keyword weight not found in Linear";

                int weightCounter = 0;
                for (long row = 0; row < linear->outputSize; ++row) {
                    for (long col = 0; col < linear->inputSize; ++col) {
                        fin.getline(buffer, sizeof(buffer), '\n');
                        linear->weight->data[weightCounter++] = (T) atof(buffer);
                    }
                }

                // Parse bias
                fin.getline(buffer, sizeof(buffer), '\n');
                if (strstr(buffer, "bias") == nullptr)
                    throw "Parse error: keyword bias not found in Linear";
                int biasCounter = 0;
                for (long i = 0; i < linear->outputSize; i++) {
                    fin.getline(buffer, sizeof(buffer), '\n');
                    linear->bias->data[biasCounter++] = (T) atof(buffer);
                }
            }
            else {
                throw "Parse error: Unknown module name";
            }

            if (module != nullptr) {
                module->next = nullptr;
                if (model == nullptr)   model = module;
                else {
                    Module<T>* current;
                    for (current = model; current->next != nullptr; current = current->next)
                        ;
                    current->next = module;
                }
            }
        }   // end of while()

        fin.close();

        return model;
    }
};


#endif //BUILD_ASCIIMODELPARSER_H
