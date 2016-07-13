//
// Created by Poter Hsu on 2015/12/11.
//

#ifndef BUILD_TYPE_H
#define BUILD_TYPE_H

enum Type {
    SPATIAL_CONVOLUTION = 1L,
    SPATIAL_MAX_POOLING = 2L,
    RELU = 3L,
    SOFTMAX = 4L,
    LOG_SOFTMAX = 5L,
    VIEW = 6L,
    DROPOUT = 7L,
    LINEAR = 8L,
    THRESHOLD = 9L
};

#endif //BUILD_TYPE_H
