//
// Created by Poter Hsu on 2015/12/25.
//

#ifndef BUILD_LINEAR_H
#define BUILD_LINEAR_H

#include "Module.hpp"
#include <cblas.h>

template <typename T>
class Linear : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    long inputSize;
    long outputSize;
    shared_ptr<Matrix<T>> weight;
    shared_ptr<Vector<T>> bias;
    shared_ptr<Vector<T>> output;

    Linear(long inputSize_, long outputSize_) : inputSize(inputSize_), outputSize(outputSize_) {
        name = "Linear";
        type = Type::LINEAR;
        weight = make_shared<Matrix<T>>(outputSize, inputSize);
        bias = make_shared<Vector<T>>(outputSize);
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim == 1);
        assert(input->nElem > 0);

        if (output == nullptr)
            output = make_shared<Vector<T>>(outputSize);

        for (long i = 0; i < outputSize; ++i) {
            output->data[i] = bias->data[i];
        }

        if (typeid(T) == typeid(double)) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        1, outputSize, inputSize, 1,
                        (double*) input->data, 1,
                        (double*) weight->data, inputSize, 1,
                        (double*) output->data, 1);
        } else {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        1, outputSize, inputSize, 1,
                        (float*) input->data, 1,
                        (float*) weight->data, inputSize, 1,
                        (float*) output->data, 1);
        }

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_LINEAR_H
