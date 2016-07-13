//
// Created by Poter Hsu on 2016/1/29.
//

#ifndef BUILD_TENSOR_H
#define BUILD_TENSOR_H

#include <vector>

using namespace std;

template <typename T>
class Tensor {
public:
    vector<long> sizes;
    vector<long> strides;
    long nDim;
    long nElem;
    T* data;

    Tensor() : sizes(vector<long>(0)), strides(vector<long>(0)), nDim(0), nElem(0), data(nullptr) {}

    Tensor(vector<long> sizes_) {
        construct(sizes_);
    }

    virtual ~Tensor() {
        if (data != nullptr)
            delete[] data;
    }

protected:
    void construct(vector<long> sizes_) {
        sizes = move(sizes_);

        size_t nDepth =  sizes.size();
        if (nDepth == 0) strides = vector<long>(0);
        else if (nDepth == 1) strides = vector<long>({0});
        else {
            // nDepth >= 2
            strides = vector<long>(nDepth - 1);
            for (size_t i = 0; i < nDepth - 1; ++i) {
                strides[i] = 1;
                for (size_t j = i + 1; j < nDepth; ++j) {
                    strides[i] *= sizes[j];
                }
            }
        }

        nDim = sizes.size();
        nElem = 1;
        for_each(begin(sizes), end(sizes), [&] (long size) {
            nElem *= size;
        });
        data = new T[nElem];
    }
};

template <typename T>
class Vector : public Tensor<T> {
public:
    long nLength;

    Vector() : Tensor<T>(), nLength(0) {}

    Vector(long nLength_) : nLength(nLength_) {
        vector<long> sizes(1);
        sizes[0] = nLength;
        Tensor<T>::construct(sizes);
    }
};

template <typename T>
class Matrix : public Tensor<T> {
public:
    long nRow;
    long nCol;
    long stride0;

    Matrix() : Tensor<T>(), nRow(0), nCol(0), stride0(0) {}

    Matrix(long nRow_, long nCol_) : nRow(nRow_), nCol(nCol_), stride0(nCol_) {
        vector<long> sizes(2);
        sizes[0] = nRow;
        sizes[1] = nCol;
        Tensor<T>::construct(sizes);
    }
};

template <typename T>
class Cube : public Tensor<T> {
public:
    long nPlane;
    long nRow;
    long nCol;
    long stride0;
    long stride1;

    Cube() : Tensor<T>(), nPlane(0), nRow(0), nCol(0), stride0(0), stride1(0) {}

    Cube(long nPlane_, long nRow_, long nCol_)
            : nPlane(nPlane_), nRow(nRow_), nCol(nCol_), stride0(nRow_ * nCol_), stride1(nCol_) {
        vector<long> sizes(3);
        sizes[0] = nPlane;
        sizes[1] = nRow;
        sizes[2] = nCol;
        Tensor<T>::construct(sizes);
    }
};

template <typename T>
class Tesseract : public Tensor<T> {
public:
    long nSpace;
    long nPlane;
    long nRow;
    long nCol;
    long stride0;
    long stride1;
    long stride2;

    Tesseract() : Tensor<T>(), nSpace(0), nPlane(0), nRow(0), nCol(0), stride0(0), stride1(0), stride2(0) {}

    Tesseract(long nSpace_, long nPlane_, long nRow_, long nCol_) :
            nSpace(nSpace_), nPlane(nPlane_), nRow(nRow_), nCol(nCol_),
            stride0(nPlane_ * nRow_ * nCol_), stride1(nRow_ * nCol_), stride2(nCol_) {
        vector<long> sizes(4);
        sizes[0] = nSpace;
        sizes[1] = nPlane;
        sizes[2] = nRow;
        sizes[3] = nCol;
        Tensor<T>::construct(sizes);
    }
};

#endif //BUILD_TENSOR_H
