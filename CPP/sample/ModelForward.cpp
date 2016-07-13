#include <iostream>
#include <opencv2/opencv.hpp>
#include "parser/AsciiModelParser.hpp"
#include "parser/BinaryModelParser.hpp"

using namespace std;
using namespace std::chrono;
using namespace cv;

//typedef double DType;
typedef float DType;

const int FaceWidth = 224;
const int FaceHeight = 224;

const double MeanB = 129.1863;
const double MeanG = 104.7624;
const double MeanR = 93.5940;

void parseModel(string parserType, string pathToModel, Module<DType>* &model) {
    if (parserType == "ascii") {
        AsciiModelParser asciiModelParser(pathToModel.c_str());
        model = asciiModelParser.parse<DType>();
    } else {
        BinaryModelParser binaryModelParser(pathToModel.c_str());
        model = binaryModelParser.parse<DType>();
    }

    if (model == nullptr) {
        cerr << "Parse model failed" << endl;
        exit(-1);
    }
}

shared_ptr<Tensor<DType>> composeInputFromImage(string pathToImage) {
    Mat img = imread(pathToImage);
    resize(img, img, Size(FaceWidth, FaceHeight));
    shared_ptr<Tensor<DType>> input = make_shared<Tensor<DType>>(vector<long>({img.channels(), img.rows, img.cols}));

    DType* pInput = input->data;
    const long inputStride0 = input->sizes[1] * input->sizes[2];

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uchar *pImageData = img.ptr(row, col);
            DType b = (DType) (pImageData[0] - MeanB);
            DType g = (DType) (pImageData[1] - MeanG);
            DType r = (DType) (pImageData[2] - MeanR);
            pInput[0 * inputStride0] = r;
            pInput[1 * inputStride0] = g;
            pInput[2 * inputStride0] = b;
            ++pInput;
        }
    }
    return input;
}

void exitWithUsageTip() {
    cerr << "Wrong argument" << endl;
    cerr << "Usage:" << endl;
    cerr << "   ModelForward ascii /path/to/Model.tpa /path/to/Image.jpg" << endl;
    cerr << "or" << endl;
    cerr << "   ModelForward binary /path/to/Model.tpb /path/to/Image.jpg" << endl;
    exit(-1);
}

int main(int argc, char* argv[]) {
    if (argc != 4)
        exitWithUsageTip();

    string parserType(argv[1]);
    string pathToModel(argv[2]);
    string pathToImage(argv[3]);

    if (parserType != "ascii" && parserType != "binary")
        exitWithUsageTip();

    Module<DType>* model;

    cout << "Parsing model ..." << endl;
    parseModel(parserType, pathToModel, model);

    cout << "Load image to input ..." << endl;
    shared_ptr<Tensor<DType>> input = composeInputFromImage(pathToImage);

    for (int i = 0; i < 2; ++i) {
        cout << "Time " << i << endl;
        cout << "  Starting forward ..." << endl;
        microseconds start = duration_cast<microseconds>(system_clock::now().time_since_epoch());
        shared_ptr<Tensor<DType>> output = model->forward(input);
        microseconds end = duration_cast<microseconds>(system_clock::now().time_since_epoch());
        double elapse = (end.count() - start.count()) / 1000000.0;
        cout << "  Elapsed: " << elapse << " (s)" << endl;

        cout << "  Output:" << endl;
        for (int i = 0; i < output->nElem; ++i) {
            cout << "    " << output->data[i] << endl;
        }
    }

    delete model;
    return 0;
}