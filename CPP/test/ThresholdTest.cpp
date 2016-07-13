//
// Created by Poter Hsu on 2015/12/25.
//

#include "ThresholdTest.h"
#include "parser/AsciiModelParser.hpp"

class ThresholdTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(ThresholdTest, NAME_CHECK) {
    Threshold<double> threshold(0);
    ASSERT_STREQ("Threshold", threshold.name);
}

TEST_F(ThresholdTest, FORWARD) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 1, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;     input->data[i++] = 0.2;   input->data[i++] = 0.3;
    input->data[i++] = -0.1;    input->data[i++] = -0.2;  input->data[i++] = -0.3;


    AsciiModelParser asciiModelParser("data/model17.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    Threshold<double>* threshold = (Threshold<double> *) model;
    ASSERT_TRUE(threshold != nullptr);

    ASSERT_NO_THROW(output = threshold->forward(input));
    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_EQ(1, output->sizes[1]);
    ASSERT_EQ(3, output->sizes[2]);
    ASSERT_NEAR(0.1, output->data[0], 1e-06);
    ASSERT_NEAR(0.2, output->data[1], 1e-06);
    ASSERT_NEAR(0.3, output->data[2], 1e-06);
    ASSERT_NEAR(0.0, output->data[3], 1e-06);
    ASSERT_NEAR(0.0, output->data[4], 1e-06);
    ASSERT_NEAR(0.0, output->data[5], 1e-06);
}