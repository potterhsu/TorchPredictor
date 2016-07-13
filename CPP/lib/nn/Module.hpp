//
// Created by Poter Hsu on 2015/12/11.
//

#ifndef BUILD_MODULE_H
#define BUILD_MODULE_H

#include <stdio.h>
#include <assert.h>
#include "Type.hpp"
#include "../tensor/Tensor.hpp"

template <typename T>
class Module {
public:
    const char* name;

    Type type;
    Module<T>* next;

    Module() {
        next = nullptr;
    }

    virtual shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) = 0;
    void list() {
        Module<T> *module = this;
        while (module) {
            printf("%s -> ", module->name);
            module = module->next;
        }
        printf("\n");
    }

    virtual ~Module() {
        if (next != nullptr) {
            delete next;
        }
    }
};


#endif //BUILD_MODULE_H
