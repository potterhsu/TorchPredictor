//
// Created by Poter Hsu on 2015/12/24.
//

#include "SoftMaxTest.h"
#include "parser/AsciiModelParser.hpp"

class SoftMaxTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(SoftMaxTest, NAME_CHECK) {
    SoftMax<double> softMax;
    ASSERT_STREQ("SoftMax", softMax.name);
}

TEST_F(SoftMaxTest, FORWARD1) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;   input->data[i++] = -0.3;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;   input->data[i++] = 0.6;

    AsciiModelParser asciiModelParser("data/model13.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SoftMax<double>* softMax = (SoftMax<double> *) model;
    ASSERT_TRUE(softMax != nullptr);

    ASSERT_NO_THROW(output = softMax->forward(input));

    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_EQ(3, output->sizes[1]);
    ASSERT_NEAR(0.36029661524054007, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 0], 1e-06);
    ASSERT_NEAR(0.39818934104493608, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 1], 1e-06);
    ASSERT_NEAR(0.24151404371452387, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 2], 1e-06);
    ASSERT_NEAR(0.30060960535572734, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 0], 1e-06);
    ASSERT_NEAR(0.33222499353334728, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 1], 1e-06);
    ASSERT_NEAR(0.36716540111092549, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 2], 1e-06);
}

TEST_F(SoftMaxTest, FORWARD2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = 144.0;    input->data[1] = -144.0;

    SoftMax<double> softMax;

    ASSERT_NO_THROW(output = softMax.forward(input));
    ASSERT_EQ(1, output->nDim);
    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_NEAR(1.0, output->data[0], 1e-06);
    ASSERT_NEAR(0.0, output->data[1], 1e-06);
}

TEST_F(SoftMaxTest, FORWARD3) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 2}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = 0.0;   input->data[1] = 1.0;
    input->data[2] = -1.0;  input->data[3] = 0.0;

    SoftMax<double> softMax;

    ASSERT_NO_THROW(output = softMax.forward(input));
    ASSERT_EQ(2, output->nDim);
    ASSERT_EQ(4, output->nElem);
    ASSERT_NEAR(0.2689414213699951, output->data[0], 1e-06);
    ASSERT_NEAR(0.7310585786300049, output->data[1], 1e-06);
    ASSERT_NEAR(0.2689414213699951, output->data[2], 1e-06);
    ASSERT_NEAR(0.7310585786300049, output->data[3], 1e-06);
}

TEST_F(SoftMaxTest, FORWARD4) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 2}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = -10.0; input->data[1] = 10.0;
    input->data[2] = 10.0;  input->data[3] = -10.0;

    SoftMax<double> softMax;

    ASSERT_NO_THROW(output = softMax.forward(input));
    ASSERT_EQ(2, output->nDim);
    ASSERT_EQ(4, output->nElem);
    ASSERT_NEAR(0.0, output->data[0], 1e-06);
    ASSERT_NEAR(1.0, output->data[1], 1e-06);
    ASSERT_NEAR(1.0, output->data[2], 1e-06);
    ASSERT_NEAR(0.0, output->data[3], 1e-06);
}