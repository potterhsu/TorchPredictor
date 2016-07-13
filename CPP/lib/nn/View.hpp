//
// Created by Poter Hsu on 2015/12/25.
//

#ifndef BUILD_VIEW_H
#define BUILD_VIEW_H

#include "Module.hpp"

template <typename T>
class View : public Module<T> {
public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    long size;
    shared_ptr<Vector<T>> output;

    View(long size_) : size(size_) {
        name = "View";
        type = Type::VIEW;
        output = nullptr;
    }
    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim > 0);
        assert(input->nElem > 0);

        const long totalSize = input->nElem;

        if (output == nullptr)
            output = make_shared<Vector<T>>(totalSize);

        if (size == -1)
            size = totalSize;
        else if (totalSize != size) {
            char message[BUFSIZ];
            sprintf(message, "Size of view is not matched, expect %ld, actual %ld", size, totalSize);
            throw length_error(message);
        }

        memcpy(output->data, input->data, sizeof(T) * totalSize);

        return next != nullptr ? next->forward(output) : output;
    }
};


#endif //BUILD_VIEW_H
