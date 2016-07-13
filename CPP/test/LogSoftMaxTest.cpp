//
// Created by Poter Hsu on 2015/12/25.
//

#include "LogSoftMaxTest.h"
#include "parser/AsciiModelParser.hpp"

class LogSoftMaxTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(LogSoftMaxTest, NAME_CHECK) {
    LogSoftMax<double> logSoftMax;
    ASSERT_STREQ("LogSoftMax", logSoftMax.name);
}

TEST_F(LogSoftMaxTest, FORWARD) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;   input->data[i++] = -0.3;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;   input->data[i++] = 0.6;

    AsciiModelParser asciiModelParser("data/model14.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    LogSoftMax<double>* logSoftMax = (LogSoftMax<double> *) model;
    ASSERT_TRUE(logSoftMax != nullptr);

    output = logSoftMax->forward(input);
    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_EQ(3, output->sizes[1]);
    ASSERT_NEAR(-1.0208276555532594, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 0], 1e-06);
    ASSERT_NEAR(-0.92082765555325929, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 1], 1e-06);
    ASSERT_NEAR(-1.4208276555532595, output->data[0 * output->sizes[0] * output->sizes[1] + 0 * output->sizes[1] + 2], 1e-06);
    ASSERT_NEAR(-1.2019428482292442, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 0], 1e-06);
    ASSERT_NEAR(-1.1019428482292439, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 1], 1e-06);
    ASSERT_NEAR(-1.0019428482292443, output->data[0 * output->sizes[0] * output->sizes[1] + 1 * output->sizes[1] + 2], 1e-06);
}