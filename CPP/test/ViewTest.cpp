//
// Created by Poter Hsu on 2015/12/25.
//

#include "ViewTest.h"
#include "parser/AsciiModelParser.hpp"

class ViewTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(ViewTest, NAME_CHECK) {
    View<double> view(0);
    ASSERT_STREQ("View", view.name);
}

TEST_F(ViewTest, FORWARD_1) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 2, 2}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;     input->data[i++] = -0.2;
    input->data[i++] = -0.01;   input->data[i++] = 0.53;

    input->data[i++] = -0.8;    input->data[i++] = 0.3;
    input->data[i++] = -0.22;   input->data[i++] = -0.3;

    View<double> view(2 * 3);

    ASSERT_THROW(output = view.forward(input), length_error);

    view.size = 2 * 2 * 2;
    ASSERT_NO_THROW(output = view.forward(input));

    ASSERT_EQ(8, output->sizes[0]);
    ASSERT_NEAR(0.1, output->data[0], 1e-06);
    ASSERT_NEAR(-0.2, output->data[1], 1e-06);
    ASSERT_NEAR(-0.01, output->data[2], 1e-06);
    ASSERT_NEAR(0.53, output->data[3], 1e-06);
    ASSERT_NEAR(-0.8, output->data[4], 1e-06);
    ASSERT_NEAR(0.3, output->data[5], 1e-06);
    ASSERT_NEAR(-0.22, output->data[6], 1e-06);
    ASSERT_NEAR(-0.3, output->data[7], 1e-06);
}

TEST_F(ViewTest, FORWARD_2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 1, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;   input->data[i++] = 0.2;   input->data[i++] = 0.3;
    input->data[i++] = -0.1;  input->data[i++] = -0.2;  input->data[i++] = -0.3;

    AsciiModelParser asciiModelParser("data/model15.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    View<double>* view = (View<double> *) model;
    ASSERT_TRUE(view != nullptr);

    ASSERT_NO_THROW(output = view->forward(input));

    ASSERT_EQ(6, output->sizes[0]);
    ASSERT_NEAR(0.1, output->data[0], 1e-06);
    ASSERT_NEAR(0.2, output->data[1], 1e-06);
    ASSERT_NEAR(0.3, output->data[2], 1e-06);
    ASSERT_NEAR(-0.1, output->data[3], 1e-06);
    ASSERT_NEAR(-0.2, output->data[4], 1e-06);
    ASSERT_NEAR(-0.3, output->data[5], 1e-06);
}

TEST_F(ViewTest, FORWARD_3) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 1, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;   input->data[i++] = 0.2;   input->data[i++] = 0.3;
    input->data[i++] = -0.1;  input->data[i++] = -0.2;  input->data[i++] = -0.3;

    AsciiModelParser asciiModelParser("data/model22.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    View<double>* view = (View<double> *) model;
    ASSERT_TRUE(view != nullptr);

    ASSERT_NO_THROW(output = view->forward(input));

    ASSERT_EQ(6, output->sizes[0]);
    ASSERT_NEAR(0.1, output->data[0], 1e-06);
    ASSERT_NEAR(0.2, output->data[1], 1e-06);
    ASSERT_NEAR(0.3, output->data[2], 1e-06);
    ASSERT_NEAR(-0.1, output->data[3], 1e-06);
    ASSERT_NEAR(-0.2, output->data[4], 1e-06);
    ASSERT_NEAR(-0.3, output->data[5], 1e-06);
}