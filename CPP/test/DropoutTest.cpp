//
// Created by Poter Hsu on 2015/12/25.
//

#include "DropoutTest.h"
#include "parser/AsciiModelParser.hpp"

class DropoutTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(DropoutTest, NAME_CHECK) {
    Dropout<double> dropout(0);
    ASSERT_STREQ("Dropout", dropout.name);
}

TEST_F(DropoutTest, FORWARD) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({2, 1, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;     input->data[i++] = 0.2;   input->data[i++] = 0.3;
    input->data[i++] = -0.1;    input->data[i++] = -0.2;  input->data[i++] = -0.3;


    AsciiModelParser asciiModelParser("data/model16.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    Dropout<double> *dropout = (Dropout<double> *) model;
    ASSERT_TRUE(dropout != nullptr);

    ASSERT_NO_THROW(output = dropout->forward(input));

    ASSERT_EQ(input->sizes[0], output->sizes[0]);
    ASSERT_EQ(input->sizes[1], output->sizes[1]);
    ASSERT_EQ(input->sizes[2], output->sizes[2]);
    ASSERT_TRUE( (output->data[0] == 0.2) || (output->data[0] == 0.0) );
    ASSERT_TRUE( (output->data[2] == 0.6) || (output->data[2] == 0.0) );
    ASSERT_TRUE( (output->data[5] == -0.6) || (output->data[5] == -0.0) );
    cout << "Dropout Log, p = 0.5" << endl;
    cout << "Input:" << endl;
    cout << input->data[0] << ", "; cout << input->data[1] << ", "; cout << input->data[2] << endl;
    cout << input->data[3] << ", "; cout << input->data[4] << ", "; cout << input->data[5] << endl;
    cout << "Output:" << endl;
    cout << output->data[0] << ", "; cout << output->data[1] << ", "; cout << output->data[2] << endl;
    cout << output->data[3] << ", "; cout << output->data[4] << ", "; cout << output->data[5] << endl;
    cout << endl;
}