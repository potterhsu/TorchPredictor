//
// Created by Poter Hsu on 2015/12/24.
//

#ifndef BUILD_SOFTMAX_H
#define BUILD_SOFTMAX_H

#include "Module.hpp"
#include <math.h>

template <typename T>
class SoftMax : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    shared_ptr<Tensor<T>> output;

    SoftMax() {
        name = "SoftMax";
        type = Type::SOFTMAX;
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim == 1 || input->nDim == 2);
        assert(input->nElem > 0);

        if (output == nullptr)
            output = make_shared<Tensor<T>>(input->sizes);

        const long inRows = input->nDim == 1 ? 1 : input->sizes[0];
        const long inCols = input->nDim == 1 ? input->sizes[0] : input->sizes[1];
        const long stride0 = input->strides[0];
        long inPlane, inRow, inCol;
        T max, v, sum;

        T* pInput = input->data;
        T* pOutput = output->data;

        for (inRow = 0; inRow < inRows; ++inRow) {
            max = -DBL_MAX;
            for (inCol = 0; inCol < inCols; ++inCol) {
                v = pInput[inCol];
                if (v > max) max = v;
            }
            for (inCol = 0; inCol < inCols; ++inCol) {
                pOutput[inCol] = exp(pInput[inCol]);
            }
            sum = 0.0;
            for (inCol = 0; inCol < inCols; ++inCol) {
                sum += pOutput[inCol];
            }
            for (inCol = 0; inCol < inCols; ++inCol) {
                v = pOutput[inCol] / sum;
                pOutput[inCol] = (v > 1) || (isnan(v)) ? 1 : v;
            }
            pInput += stride0;
            pOutput += stride0;
        }

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_SOFTMAX_H
