//
// Created by Poter Hsu on 2015/12/11.
//

#ifndef BUILD_SPATIALCONVOLUTION_H
#define BUILD_SPATIALCONVOLUTION_H

#include "Module.hpp"
#include <cblas.h>

template <typename T>
class SpatialConvolution : public Module<T> {
private:
    unique_ptr<Matrix<T>> jointInput;

public:
    using Module<T>::name;
    using Module<T>::type;
    using Module<T>::next;
    long nInputPlane;
    long nOutputPlane;
    long kW;
    long kH;
    long dW;
    long dH;
    long padW;
    long padH;
    shared_ptr<Tesseract<T>> weight;
    shared_ptr<Vector<T>> bias;
    shared_ptr<Cube<T>> output;

    SpatialConvolution(long nInputPlane_, long nOutputPlane_, long kW_, long kH_, long dW_, long dH_, long padW_, long padH_)
            : nInputPlane(nInputPlane_), nOutputPlane(nOutputPlane_), kW(kW_), kH(kH_), dW(dW_), dH(dH_), padW(padW_), padH(padH_) {
        name = "SpatialConvolution";
        type = Type::SPATIAL_CONVOLUTION;
        weight = make_shared<Tesseract<T>>(nOutputPlane, nInputPlane, kH, kW);
        bias = make_shared<Vector<T>>(nOutputPlane);
        jointInput = nullptr;
        output = nullptr;
    }

    shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) override {
        assert(input->nDim == 3);
        assert(input->nElem > 0);

        const long kSize = kW * kH;
        const long inRows = input->sizes[1];
        const long inCols = input->sizes[2];
        const long outRows = (inRows + 2 * padH - kH) / dH + 1;
        const long outCols = (inCols + 2 * padW - kW) / dW + 1;
        const long jointInputRows = nInputPlane * kSize;
        const long jointInputCols = outRows * outCols;
        const long jointKernelRows = nOutputPlane;
        const long jointKernelCols = nInputPlane * kSize;
        const long jointOutputCols = jointInputCols;
        long inPlane, inRow, inCol, outPlane, outRow, outCol, kx, ky;

        /* jointInput
         * -------------------- */
        if (jointInput == nullptr)
            jointInput = unique_ptr<Matrix<T>>(new Matrix<T>(jointInputRows, jointInputCols));
        T* pInput = input->data;
        T* pJointInput0 = jointInput->data;
        T* pJointInput1;
        T* pJointInput2;
        T* pJointInput3;
        T v;
        for (inPlane = 0; inPlane < nInputPlane; ++inPlane) {
            pJointInput1 = pJointInput0;
            for (inRow = -padH; inRow <= inRows - kH + padH; inRow += dH) {
                pJointInput2 = pJointInput1;
                for (ky = 0; ky < kH; ++ky) {
                    pJointInput3 = pJointInput2;
                    if (inRow + ky < 0 || inRow + ky >= inRows) {
                        for (inCol = -padW; inCol <= inCols - kW + padW; inCol += dW) {
                            for (kx = 0; kx < kW; ++kx) {
                                pJointInput3[kx * jointInput->stride0] = 0.0;
                            }
                            ++pJointInput3;
                        }
                    } else {
                        for (inCol = -padW; inCol <= inCols - kW + padW; inCol += dW) {
                            for (kx = 0; kx < kW; ++kx) {
                                if (inCol + kx < 0 || inCol + kx >= inCols) v = 0.0;
                                else v = pInput[(inRow + ky) * input->strides[1] + inCol + kx];
                                pJointInput3[kx * jointInput->stride0] = v;
                            }
                            ++pJointInput3;
                        }
                    }
                    pJointInput2 += kW * jointInput->stride0;
                }
                pJointInput1 += outCols;
            }
            pJointInput0 += kSize * jointInput->stride0;
            pInput += input->strides[0];
        }

        /* jointKernel
         * -------------------- */
        T* jointKernel = weight->data;

        /* output
         * -------------------- */
        if (output == nullptr)
            output = make_shared<Cube<T>>(nOutputPlane, outRows, outCols);
        T* pBias = bias->data;
        T* jointOutput = output->data;
        T* pJointOutput = jointOutput;
        for (outPlane = 0; outPlane < nOutputPlane; ++outPlane) {
            for (outRow = 0; outRow < outRows; ++outRow) {
                for (outCol = 0; outCol < outCols; ++outCol) {
                    *pJointOutput++ = *pBias;
                }
            }
            ++pBias;
        }

        if (typeid(T) == typeid(double)) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        jointInputCols, jointKernelRows, jointInputRows, 1,
                        (double *) jointInput->data, jointInputCols,
                        (double *) jointKernel, jointKernelCols, 1,
                        (double *) jointOutput, jointOutputCols);
        } else {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        jointInputCols, jointKernelRows, jointInputRows, 1,
                        (float *) jointInput->data, jointInputCols,
                        (float *) jointKernel, jointKernelCols, 1,
                        (float *) jointOutput, jointOutputCols);
        }

        return next != nullptr ? next->forward(output) : output;
    }
};

#endif //BUILD_SPATIALCONVOLUTION_H
