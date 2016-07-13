//
// Created by Poter Hsu on 2015/12/25.
//

#ifndef BUILD_DROPOUT_H
#define BUILD_DROPOUT_H

#include "Module.hpp"

template <typename T>
class Dropout : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    float p;
    shared_ptr<Tensor<T>> output;

    Dropout(float p_) : p(p_) {
        name = "Dropout";
        type = Type::DROPOUT;
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim > 0);
        assert(input->nElem > 0);
        assert(p >= 0.0f && p <= 1.0f);

        if (output == nullptr)
            output = make_shared<Tensor<T>>(input->sizes);

        const float scale = 1.0f / (1.0f - p);

        T* pInput = input->data;
        T* pOutput = output->data;

        for (long i = 0; i < input->nElem; ++i) {
            *pOutput = (float(arc4random()) / float(UINT32_MAX)) < p ? 0.0 : (*pInput * scale);
            ++pInput;
            ++pOutput;
        }

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_DROPOUT_H
