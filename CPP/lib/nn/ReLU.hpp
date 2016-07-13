//
// Created by Poter Hsu on 2015/12/23.
//

#ifndef BUILD_RELU_H
#define BUILD_RELU_H

#include "Module.hpp"

template <typename T>
class ReLU : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    shared_ptr<Tensor<T>> output;

    ReLU() {
        name = "ReLU";
        type = Type::RELU;
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim > 0);
        assert(input->nElem > 0);

        if (output == nullptr)
            output = make_shared<Tensor<T>>(input->sizes);

        T* pInput = input->data;
        T* pOutput = output->data;

        for (long i = 0; i < output->nElem; ++i) {
            *pOutput = *pInput < 0.0 ? 0.0 : *pInput;
            ++pInput;
            ++pOutput;
        }

        return next != nullptr ? next->forward(output) : output;
    }
};

#endif //BUILD_RELU_H
