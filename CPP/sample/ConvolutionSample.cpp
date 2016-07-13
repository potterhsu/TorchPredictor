#include <iostream>
#include <chrono>
#include "parser/AsciiModelParser.hpp"

using namespace std;
using namespace std::chrono;

typedef double DType;
//typedef float DType;

int main() {
    shared_ptr<Cube<DType>> input = make_shared<Cube<DType>>(3, 256, 256);
    shared_ptr<Tensor<DType>> output;
    for (long inPlane = 0; inPlane < input->nPlane; ++inPlane) {
        for (long r = 0; r < input->nRow; ++r) {
            for (long c = 0; c < input->nCol; ++c) {
                input->data[inPlane * input->nRow * input->nCol + r * input->nCol + c] = arc4random() / DType(UINT32_MAX);
            }
        }
    }

    SpatialConvolution<DType>* spatialConvolution = new SpatialConvolution<DType>(3, 64, 3, 3, 1, 1, 1, 1);
    for (long n = 0; n < spatialConvolution->weight->nElem; ++n) {
        spatialConvolution->weight->data[n] = arc4random() / DType(UINT32_MAX);
    }

    double total = 0;
    int times = 20;
    for (int i = 0; i < times; ++i) {
        microseconds start = duration_cast<microseconds>(system_clock::now().time_since_epoch());
        output = spatialConvolution->forward(input);
        microseconds end = duration_cast<microseconds>(system_clock::now().time_since_epoch());
        double elapse = (end.count() - start.count()) / 1000000.0;
        total += elapse;
        cout << "Convolution elapsed: " << elapse << " (s)" << endl;
    }
    cout << "Avg: " << total / times << " (s)" << endl;

    return 0;
}