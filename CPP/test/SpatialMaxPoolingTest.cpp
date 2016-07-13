//
// Created by Poter Hsu on 2015/12/22.
//

#include "SpatialMaxPoolingTest.h"
#include "parser/AsciiModelParser.hpp"

class SpatialMaxPoolingTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(SpatialMaxPoolingTest, NAME_CHECK) {
    SpatialMaxPooling<double> spatialMaxPooling(0, 0, 0, 0, 0, 0);
    ASSERT_STREQ("SpatialMaxPooling", spatialMaxPooling.name);
}

TEST_F(SpatialMaxPoolingTest, FORWARD_kW1kH1dW1dH1padW0padH0_inputW3H3_outputW3H3) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({1, 3, 3}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = 0.1;    input->data[1] = 0.2;    input->data[2] = 0.3;
    input->data[3] = 0.4;    input->data[4] = 0.5;    input->data[5] = 0.6;
    input->data[6] = 0.7;    input->data[7] = 0.8;    input->data[8] = 0.9;

    AsciiModelParser asciiModelParser("data/model8.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialMaxPooling<double>* spatialMaxPooling = (SpatialMaxPooling<double> *) model;
    ASSERT_TRUE(spatialMaxPooling != nullptr);

    output = spatialMaxPooling->forward(input);
    ASSERT_EQ(1, output->sizes[0]);
    ASSERT_NEAR(0.1, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.5, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.9, output->data[0 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 2], 1e-06);
}

TEST_F(SpatialMaxPoolingTest, FORWARD_kW3kH3dW2dH2padW1padH1_inputW5H3_outputW3H2) {
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

    AsciiModelParser asciiModelParser("data/model9.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialMaxPooling<double>* spatialMaxPooling = (SpatialMaxPooling<double> *) model;
    ASSERT_TRUE(spatialMaxPooling != nullptr);

    output = spatialMaxPooling->forward(input);
    ASSERT_EQ(5, output->sizes[0]);
    ASSERT_NEAR(0.5, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.8, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.4, output->data[4 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.6, output->data[4 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.8, output->data[4 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 2], 1e-06);
}

TEST_F(SpatialMaxPoolingTest, FORWARD_CONV_MAXPOOLING) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 5, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;
    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;
    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;
    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;
    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;
    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;

    AsciiModelParser asciiModelParser("data/model10.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);
    ASSERT_TRUE(spatialConvolution->next != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_EQ(5, output->sizes[0]);
    ASSERT_NEAR(0.44297416575738519, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.44297416575738519, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.29523706801510019, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.29523706801510019, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.11224558669241141, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.050492804082287235, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.29107464728382232, output->data[3 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.29107464728382232, output->data[3 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.077492991110475584, output->data[4 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.029402459251089717, output->data[4 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
}
