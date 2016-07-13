#include <iostream>
#include <fstream>
#include <unistd.h>
#include "parser/AsciiModelParser.hpp"

using namespace std;

bool inline isFileExist(const string& filename);
void startConvert(ifstream& fin, ofstream& fout);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Wrong argument" << endl;
        cerr << "Usage: Ascii2BinaryModelConverter /path/to/AsciiModel.tpa /path/to/BinaryModel.tpb" << endl;
        return -1;
    }

    string pathToAsciiModel(argv[1]);
    string pathToBinaryModel(argv[2]);

    if (isFileExist(pathToBinaryModel)) {
        cerr << pathToBinaryModel << " is already exist." << endl;
        return -1;
    }

    ifstream fin(pathToAsciiModel, ios::in);
    if (!fin) {
        cerr << "Cannot open " << pathToAsciiModel << endl;
        return -1;
    }

    ofstream fout(pathToBinaryModel, ios::out | ios::binary);
    if (!fout) {
        cerr << "Cannot access " << pathToBinaryModel << endl;
        return -1;
    }

    cout << "Processing..." << endl;

    try {
        startConvert(fin, fout);
    } catch (const char* msg) {
        cerr << msg << endl;
        return -1;
    }

    cout << "Done" << endl;

    fin.close();
    fout.close();

    return 0;
}

bool inline isFileExist(const string& filename) {
    return access(filename.c_str(), F_OK) != -1;
}

void startConvert(ifstream& fin, ofstream& fout) {
    char buffer[BUFSIZ];
    long lvalue;
    double dvalue;

    while (fin.getline(buffer, sizeof(buffer), '\n')) {
        if (strstr(buffer, "nn.Sequential") != nullptr) {
            // Do nothing
        }
        else if (strstr(buffer, "nn.SpatialConvolution") != nullptr) {
            lvalue = Type::SPATIAL_CONVOLUTION;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse nInputPlane
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "nInputPlane") == nullptr)
                throw "Parse error: keyword nInputPlane not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long nInputPlane = lvalue;

            // Parse nOutputPlane
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "nOutputPlane") == nullptr)
                throw "Parse error: keyword nOutputPlane not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long nOutputPlane = lvalue;

            // Parse kW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "kW") == nullptr)
                throw "Parse error: keyword kW not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long kW = lvalue;

            // Parse kH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "kH") == nullptr)
                throw "Parse error: keyword kH not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long kH = lvalue;

            // Parse dW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "dW") == nullptr)
                throw "Parse error: keyword dW not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse dH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "dH") == nullptr)
                throw "Parse error: keyword dH not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse padW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "padW") == nullptr)
                throw "Parse error: keyword padW not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse padH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "padH") == nullptr)
                throw "Parse error: keyword padH not found in SpatialConvolution";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse weight
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "weight") == nullptr)
                throw "Parse error: keyword weight not found in SpatialConvolution";
            for (long outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                for (long inPlane = 0; inPlane < nInputPlane; ++inPlane) {
                    for (long row = 0; row < kH; ++row) {
                        for (long col = 0; col < kW; ++col) {
                            fin.getline(buffer, sizeof(buffer), '\n');
                            dvalue = atof(buffer);
                            fout.write((char *) &dvalue, sizeof(dvalue));
                        }
                    }
                }
            }

            // Parse bias
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "bias") == nullptr)
                throw "Parse error: keyword bias not found in SpatialConvolution";
            for (int outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
                fin.getline(buffer, sizeof(buffer), '\n');
                dvalue = atof(buffer);
                fout.write((char *) &dvalue, sizeof(dvalue));
            }
        }
        else if (strstr(buffer, "nn.SpatialMaxPooling") != nullptr) {
            lvalue = Type::SPATIAL_MAX_POOLING;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse kW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "kW") == nullptr)
                throw "Parse error: keyword kW not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse kH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "kH") == nullptr)
                throw "Parse error: keyword kH not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse dW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "dW") == nullptr)
                throw "Parse error: keyword dW not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse dH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "dH") == nullptr)
                throw "Parse error: keyword dH not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse padW
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "padW") == nullptr)
                throw "Parse error: keyword padW not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse padH
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "padH") == nullptr)
                throw "Parse error: keyword padH not found in SpatialMaxPooling";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
        }
        else if (strstr(buffer, "nn.ReLU") != nullptr) {
            lvalue = Type::RELU;
            fout.write((char *) &lvalue, sizeof(lvalue));
        }
        else if (strstr(buffer, "nn.SoftMax") != nullptr) {
            lvalue = Type::SOFTMAX;
            fout.write((char *) &lvalue, sizeof(lvalue));
        }
        else if (strstr(buffer, "nn.LogSoftMax") != nullptr) {
            lvalue = Type::LOG_SOFTMAX;
            fout.write((char *) &lvalue, sizeof(lvalue));
        }
        else if (strstr(buffer, "nn.View") != nullptr) {
            lvalue = Type::VIEW;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse size
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "size") == nullptr)
                throw "Parse error: keyword size not found in View";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
        }
        else if (strstr(buffer, "nn.Dropout") != nullptr) {
            lvalue = Type::DROPOUT;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse p
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "p") == nullptr)
                throw "Parse error: keyword p not found in Dropout";
            fin.getline(buffer, sizeof(buffer), '\n');
            dvalue = atof(buffer);
            fout.write((char *) &dvalue, sizeof(dvalue));
        }
        else if (strstr(buffer, "nn.Threshold") != nullptr) {
            lvalue = Type::THRESHOLD;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse threshold
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "threshold") == nullptr)
                throw "Parse error: keyword threshold not found in Threshold";
            fin.getline(buffer, sizeof(buffer), '\n');
            dvalue = atof(buffer);
            fout.write((char *) &dvalue, sizeof(dvalue));
        }
        else if (strstr(buffer, "nn.Linear") != nullptr) {
            lvalue = Type::LINEAR;
            fout.write((char *) &lvalue, sizeof(lvalue));

            // Parse inputSize
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "inputSize") == nullptr)
                throw "Parse error: keyword inputSize not found in Linear";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long inputSize = lvalue;

            // Parse outputSize
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "outputSize") == nullptr)
                throw "Parse error: keyword outputSize not found in Linear";
            fin.getline(buffer, sizeof(buffer), '\n');
            lvalue = (long) atoi(buffer);
            fout.write((char *) &lvalue, sizeof(lvalue));
            long outputSize = lvalue;

            // Parse weight
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "weight") == nullptr)
                throw "Parse error: keyword weight not found in Linear";

            for (long row = 0; row < outputSize; ++row) {
                for (long col = 0; col < inputSize; ++col) {
                    fin.getline(buffer, sizeof(buffer), '\n');
                    dvalue = atof(buffer);
                    fout.write((char *) &dvalue, sizeof(dvalue));
                }
            }
            // Parse bias
            fin.getline(buffer, sizeof(buffer), '\n');
            if (strstr(buffer, "bias") == nullptr)
                throw "Parse error: keyword bias not found in Linear";
            for (long i = 0; i < outputSize; i++) {
                fin.getline(buffer, sizeof(buffer), '\n');
                dvalue = atof(buffer);
                fout.write((char *) &dvalue, sizeof(dvalue));
            }
        }
        else {
            throw "Parse error: Unknown module name";
        }
    }   // end of while()
}