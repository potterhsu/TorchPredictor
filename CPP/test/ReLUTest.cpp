//
// Created by Poter Hsu on 2015/12/23.
//

#include "ReLUTest.h"
#include "parser/AsciiModelParser.hpp"

class ReLUTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(ReLUTest, NAME_CHECK) {
    ReLU<double> reLU;
    ASSERT_STREQ("ReLU", reLU.name);
}

TEST_F(ReLUTest, FORWARD) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 2, 2}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = -0.2;
    input->data[i++] = -0.01;  input->data[i++] = 0.53;
    
    input->data[i++] = -0.8;    input->data[i++] = 0.3;
    input->data[i++] = -0.22;   input->data[i++] = -0.3;

    AsciiModelParser asciiModelParser("data/model11.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    ReLU<double>* reLU = (ReLU<double> *) model;
    ASSERT_TRUE(reLU != nullptr);

    output = reLU->forward(input);
    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_NEAR(0.1, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.0, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.0, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.53, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.0, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.3, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.0, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.0, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
}