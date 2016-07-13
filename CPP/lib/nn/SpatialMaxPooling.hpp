//
// Created by Poter Hsu on 2015/12/22.
//

#ifndef BUILD_SPATIALMAXPOOLING_H
#define BUILD_SPATIALMAXPOOLING_H

#include "Module.hpp"
#include <float.h>

template <typename T>
class SpatialMaxPooling : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    long kW;
    long kH;
    long dW;
    long dH;
    long padW;
    long padH;
    shared_ptr<Cube<T>> output;

    SpatialMaxPooling(long kW_, long kH_, long dW_, long dH_, long padW_, long padH_)
            : kW(kW_), kH(kH_), dW(dW_), dH(dH_), padW(padW_), padH(padH_) {
        name = "SpatialMaxPooling";
        type = Type::SPATIAL_MAX_POOLING;
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim == 3);
        assert(input->nElem > 0);

        const long inPlanes = input->sizes[0];
        const long inRows = input->sizes[1];
        const long inCols = input->sizes[2];
        const long outPlanes = inPlanes;
        const long outRows = (inRows + 2 * padH - kH) / dH + 1;
        const long outCols = (inCols + 2 * padW - kW) / dW + 1;
        const long m = inRows - kH + padH;
        const long n = inCols - kW + padW;
        const long p = inRows - kH;
        const long q = inCols - kW;
        long inPlane, inRow, inCol, kx, ky;
        T max, v;

        if (output == nullptr)
            output = make_shared<Cube<T>>(outPlanes, outRows, outCols);

        T* pOutput = output->data;
        T* pInput = input->data;

        for (inPlane = 0; inPlane < inPlanes; ++inPlane) {
            for (inRow = -padH; inRow <= m; inRow += dH) {
                for (inCol = -padW; inCol <= n; inCol += dW) {
                    max = -DBL_MAX;
                    if (inRow < 0 || inCol < 0 || inRow > p || inCol > q) {
                        for (ky = 0; ky < kH; ++ky) {
                            if (inRow + ky < 0 || inRow + ky >= inRows) continue;
                            for (kx = 0; kx < kW; ++kx) {
                                if (inCol + kx < 0 || inCol + kx >= inCols) continue;
                                v = pInput[(inRow + ky) * input->strides[1] + inCol + kx];
                                if (v > max) max = v;
                            }
                        }
                    }
                    else {
                        for (ky = 0; ky < kH; ++ky) {
                            for (kx = 0; kx < kW; ++kx) {
                                v = pInput[(inRow + ky) * input->strides[1] + inCol + kx];
                                if (v > max) max = v;
                            }
                        }
                    }
                    *pOutput++ = max;
                }
            }
            pInput += input->strides[0];
        }

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_SPATIALMAXPOOLING_H
