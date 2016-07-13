//
// Created by Poter Hsu on 2015/12/25.
//

#include "LinearTest.h"
#include "parser/AsciiModelParser.hpp"

class LinearTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(LinearTest, NAME_CHECK) {
    Linear<double> linear(0, 0);
    ASSERT_STREQ("Linear", linear.name);
}

TEST_F(LinearTest, FORWARD) {
    shared_ptr<Vector<double>> input = make_shared<Vector<double>>(3);
    shared_ptr<Tensor<double>> output;

    input->data[0] = 0.1;    input->data[1] = 0.2;   input->data[2] = 0.3;


    AsciiModelParser asciiModelParser("data/model18.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    Linear<double> *linear = (Linear<double> *) model;
    ASSERT_TRUE(linear != nullptr);

    ASSERT_NO_THROW(output = linear->forward(input));
    ASSERT_EQ(2, output->sizes[0]);
    ASSERT_NEAR(-0.45340796869971406, output->data[0], 1e-06);
    ASSERT_NEAR(-0.51817160589954137, output->data[1], 1e-06);
}