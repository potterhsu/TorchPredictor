//
// Created by Poter Hsu on 2016/3/1.
//

#include "BinaryModelParserTest.h"
#include "parser/BinaryModelParser.hpp"
#include <opencv2/opencv.hpp>

using namespace chrono;
using namespace cv;

class BinaryModelParserTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(BinaryModelParserTest, PARSE_NON_EXIST_MODEL_SHOULD_THROW_EXCEPTION) {
    BinaryModelParser binaryModelParser("non-exist-model.tpb");
    ASSERT_THROW(model = binaryModelParser.parse<double>(), runtime_error);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_1) {
    BinaryModelParser binaryModelParser("data/model1.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double> *spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);
    ASSERT_STREQ("SpatialConvolution", spatialConvolution->name);
    ASSERT_EQ(1, spatialConvolution->nInputPlane);
    ASSERT_EQ(2, spatialConvolution->nOutputPlane);
    ASSERT_EQ(2, spatialConvolution->kW);
    ASSERT_EQ(2, spatialConvolution->kH);
    ASSERT_EQ(1, spatialConvolution->dW);
    ASSERT_EQ(1, spatialConvolution->dH);
    ASSERT_EQ(0, spatialConvolution->padW);
    ASSERT_EQ(0, spatialConvolution->padH);
    ASSERT_EQ(spatialConvolution->nOutputPlane, spatialConvolution->weight->nSpace);
    ASSERT_EQ(spatialConvolution->nInputPlane, spatialConvolution->weight->nPlane);
    ASSERT_EQ(spatialConvolution->kH, spatialConvolution->weight->nRow);
    ASSERT_EQ(spatialConvolution->kW, spatialConvolution->weight->nCol);
    ASSERT_NEAR(0.0095892953686416, spatialConvolution->weight->data[0], 1e-06);
    ASSERT_EQ(spatialConvolution->nOutputPlane, spatialConvolution->bias->nLength);
    ASSERT_NEAR(-0.35622525960207, spatialConvolution->bias->data[0], 1e-06);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_2) {
    BinaryModelParser binaryModelParser("data/model2.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);
    ASSERT_STREQ("SpatialConvolution", spatialConvolution->name);
    ASSERT_EQ(3, spatialConvolution->nInputPlane);
    ASSERT_EQ(5, spatialConvolution->nOutputPlane);
    ASSERT_EQ(2, spatialConvolution->kW);
    ASSERT_EQ(2, spatialConvolution->kH);
    ASSERT_EQ(1, spatialConvolution->dW);
    ASSERT_EQ(1, spatialConvolution->dH);
    ASSERT_EQ(0, spatialConvolution->padW);
    ASSERT_EQ(0, spatialConvolution->padH);
    ASSERT_EQ(spatialConvolution->nOutputPlane, spatialConvolution->weight->nSpace);
    ASSERT_EQ(spatialConvolution->nInputPlane, spatialConvolution->weight->nPlane);
    ASSERT_EQ(spatialConvolution->kH, spatialConvolution->weight->nRow);
    ASSERT_EQ(spatialConvolution->kW, spatialConvolution->weight->nCol);
    ASSERT_NEAR(0.25797212822465, spatialConvolution->weight->data[0], 1e-06);
    ASSERT_EQ(spatialConvolution->nOutputPlane, spatialConvolution->bias->nLength);
    ASSERT_NEAR(-0.14751877819581, spatialConvolution->bias->data[0], 1e-06);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_3) {
    BinaryModelParser binaryModelParser("data/model8.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialMaxPooling<double> *spatialMaxPooling = (SpatialMaxPooling<double> *) model;
    ASSERT_TRUE(spatialMaxPooling != nullptr);
    ASSERT_STREQ("SpatialMaxPooling", spatialMaxPooling->name);
    ASSERT_EQ(1, spatialMaxPooling->kW);
    ASSERT_EQ(1, spatialMaxPooling->kH);
    ASSERT_EQ(1, spatialMaxPooling->dW);
    ASSERT_EQ(1, spatialMaxPooling->dH);
    ASSERT_EQ(0, spatialMaxPooling->padW);
    ASSERT_EQ(0, spatialMaxPooling->padH);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_4) {
    BinaryModelParser binaryModelParser("data/model10.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double> *spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);
    ASSERT_STREQ("SpatialConvolution", spatialConvolution->name);

    SpatialMaxPooling<double>* spatialMaxPooling = (SpatialMaxPooling<double> *) spatialConvolution->next;
    ASSERT_TRUE(spatialMaxPooling != nullptr);
    ASSERT_STREQ("SpatialMaxPooling", spatialMaxPooling->name);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_5) {
    BinaryModelParser binaryModelParser("data/model11.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    ReLU<double>* reLU = (ReLU<double> *) model;
    ASSERT_TRUE(reLU != nullptr);
    ASSERT_STREQ("ReLU", reLU->name);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_6) {
    BinaryModelParser binaryModelParser("data/model12.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);
    ASSERT_STREQ("SpatialConvolution", spatialConvolution->name);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_7) {
    BinaryModelParser binaryModelParser("data/model13.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SoftMax<double>* softMax = (SoftMax<double> *) model;
    ASSERT_TRUE(softMax != nullptr);
    ASSERT_STREQ("SoftMax", softMax->name);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_8) {
    BinaryModelParser binaryModelParser("data/model14.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    LogSoftMax<double>* logSoftMax = (LogSoftMax<double> *) model;
    ASSERT_TRUE(logSoftMax != nullptr);
    ASSERT_STREQ("LogSoftMax", logSoftMax->name);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_9) {
    BinaryModelParser binaryModelParser("data/model15.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    View<double>* view = (View<double> *) model;
    ASSERT_TRUE(view != nullptr);
    ASSERT_STREQ("View", view->name);
    ASSERT_EQ(6, view->size);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_10) {
    BinaryModelParser binaryModelParser("data/model16.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    Dropout<double>* dropout = (Dropout<double> *) model;
    ASSERT_TRUE(dropout != nullptr);
    ASSERT_STREQ("Dropout", dropout->name);
    ASSERT_FLOAT_EQ(0.5, dropout->p);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_11) {
    BinaryModelParser binaryModelParser("data/model17.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    Threshold<double>* threshold = (Threshold<double> *) model;
    ASSERT_TRUE(threshold != nullptr);
    ASSERT_STREQ("Threshold", threshold->name);
    ASSERT_NEAR(0.00000100, threshold->threshold, 1e-06);
}

TEST_F(BinaryModelParserTest, PARSE_CHECK_12) {
    BinaryModelParser binaryModelParser("data/model18.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    Linear<double>* linear = (Linear<double> *) model;
    ASSERT_TRUE(linear != nullptr);
    ASSERT_STREQ("Linear", linear->name);
    ASSERT_EQ(3, linear->inputSize);
    ASSERT_EQ(2, linear->outputSize);
    ASSERT_NEAR(0.01639092608092, linear->weight->data[0 * linear->weight->nCol + 0], 1e-06);
    ASSERT_NEAR(0.33697351665737, linear->weight->data[0 * linear->weight->nCol + 1], 1e-06);
    ASSERT_NEAR(-0.029063054975567, linear->weight->data[0 * linear->weight->nCol + 2], 1e-06);
    ASSERT_NEAR(-0.036998662712144, linear->weight->data[1 * linear->weight->nCol + 0], 1e-06);
    ASSERT_NEAR(-0.36142456362889, linear->weight->data[1 * linear->weight->nCol + 1], 1e-06);
    ASSERT_NEAR(-0.18486751288563, linear->weight->data[1 * linear->weight->nCol + 2], 1e-06);
    ASSERT_NEAR(-0.51372284814661, linear->bias->data[0], 1e-06);
    ASSERT_NEAR(-0.38672657303686, linear->bias->data[1], 1e-06);
}

TEST_F(BinaryModelParserTest, FORWARD_1) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = -0.1;   input->data[i++] = -0.2;   input->data[i++] = -0.3;   input->data[i++] = -0.4;   input->data[i++] = -0.5;
    input->data[i++] = -0.4;   input->data[i++] = -0.5;   input->data[i++] = -0.6;   input->data[i++] = -0.7;   input->data[i++] = -0.8;
    input->data[i++] = -0.7;   input->data[i++] = -0.8;   input->data[i++] = -0.9;   input->data[i++] = -0.1;   input->data[i++] = -0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = -0.1;   input->data[i++] = 0.2;    input->data[i++] = -0.3;   input->data[i++] = 0.4;    input->data[i++] = -0.5;
    input->data[i++] = 0.4;    input->data[i++] = -0.5;   input->data[i++] = 0.6;    input->data[i++] = -0.7;   input->data[i++] = 0.8;
    input->data[i++] = -0.7;   input->data[i++] = 0.8;    input->data[i++] = -0.9;   input->data[i++] = 0.1;    input->data[i++] = -0.2;


    BinaryModelParser binaryModelParser("data/model19.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    ASSERT_NO_THROW(output = spatialConvolution->forward(input));
    ASSERT_EQ(50, output->nElem);
    ASSERT_NEAR(0.031050691124081459, output->data[0], 1e-06);
    ASSERT_NEAR(-0.17780420225856872, output->data[1], 1e-06);
    ASSERT_NEAR(-0.070620533069490107, output->data[48], 1e-06);
    ASSERT_NEAR(0.46464632326899608, output->data[49], 1e-06);
}

TEST_F(BinaryModelParserTest, FORWARD_2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = -0.1;   input->data[i++] = -0.2;   input->data[i++] = -0.3;   input->data[i++] = -0.4;   input->data[i++] = -0.5;
    input->data[i++] = -0.4;   input->data[i++] = -0.5;   input->data[i++] = -0.6;   input->data[i++] = -0.7;   input->data[i++] = -0.8;
    input->data[i++] = -0.7;   input->data[i++] = -0.8;   input->data[i++] = -0.9;   input->data[i++] = -0.1;   input->data[i++] = -0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = -0.1;   input->data[i++] = 0.2;    input->data[i++] = -0.3;   input->data[i++] = 0.4;    input->data[i++] = -0.5;
    input->data[i++] = 0.4;    input->data[i++] = -0.5;   input->data[i++] = 0.6;    input->data[i++] = -0.7;   input->data[i++] = 0.8;
    input->data[i++] = -0.7;   input->data[i++] = 0.8;    input->data[i++] = -0.9;   input->data[i++] = 0.1;    input->data[i++] = -0.2;


    BinaryModelParser binaryModelParser("data/model20.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    ASSERT_NO_THROW(output = spatialConvolution->forward(input));
    ASSERT_EQ(100, output->nElem);
    ASSERT_NEAR(-4.5197470635355623, output->data[0], 1e-06);
    ASSERT_NEAR(-4.4662822965703048, output->data[1], 1e-06);
    ASSERT_NEAR(-4.6526244798414362, output->data[98], 1e-06);
    ASSERT_NEAR(-4.5594355160327131, output->data[99], 1e-06);
}

TEST_F(BinaryModelParserTest, FORWARD_3) {
    Mat img = imread("data/lena.jpg");

    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({img.channels(), img.rows, img.cols}));
    shared_ptr<Tensor<double>> output;

    double* pInput = input->data;
    const long inputStride0 = input->sizes[1] * input->sizes[2];

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uchar *pImageData = img.ptr(row, col);
            double b = double(pImageData[0]) / 255.0;
            double g = double(pImageData[1]) / 255.0;
            double r = double(pImageData[2]) / 255.0;
            pInput[0 * inputStride0] = r;
            pInput[1 * inputStride0] = g;
            pInput[2 * inputStride0] = b;
            ++pInput;
        }
    }

    BinaryModelParser binaryModelParser("data/model21.tpb");
    ASSERT_NO_THROW(model = binaryModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    microseconds start = duration_cast<microseconds>(system_clock::now().time_since_epoch());
    ASSERT_NO_THROW(output = spatialConvolution->forward(input));
    microseconds end = duration_cast<microseconds>(system_clock::now().time_since_epoch());
    double elapse = (end.count() - start.count()) / 1000000.0;
    cout << "BinaryModelParserTest.FORWARD_3 elapsed: " << elapse << " (s)" << endl << endl;

    ASSERT_EQ(1, output->nDim);
    ASSERT_EQ(4, output->nElem);
    ASSERT_NEAR(-1.3603096089503102, output->data[0], 1e-06);
    ASSERT_NEAR(-1.3913380783825999, output->data[1], 1e-06);
    ASSERT_NEAR(-1.4282591609878621, output->data[2], 1e-06);
    ASSERT_NEAR(-1.3666856874854048, output->data[3], 1e-06);
}