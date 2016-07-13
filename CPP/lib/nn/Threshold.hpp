//
// Created by Poter Hsu on 2015/12/25.
//

#ifndef BUILD_THRESHOLD_H
#define BUILD_THRESHOLD_H

#include "Module.hpp"

template <typename T>
class Threshold : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    T threshold;
    shared_ptr<Tensor<T>> output;

    Threshold(T threshold_) : threshold(threshold_) {
        name = "Threshold";
        type = Type::THRESHOLD;
        output = nullptr;
    }
    
    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim > 0);
        assert(input->nElem > 0);

        if (output == nullptr)
            output = make_shared<Tensor<T>>(input->sizes);

        T* pInput = input->data;
        T* pOutput = output->data;

        for (long i = 0; i < input->nElem; ++i) {
            *pOutput = *pInput < 0 ? 0.0 : *pInput;
            ++pInput;
            ++pOutput;
        }

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_THRESHOLD_H
