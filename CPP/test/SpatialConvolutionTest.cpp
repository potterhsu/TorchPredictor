//
// Created by Poter Hsu on 2015/12/18.
//

#include "SpatialConvolutionTest.h"
#include "parser/AsciiModelParser.hpp"

class SpatialConvolutionTest : public testing::Test {
protected:
    Module<double>* model;

    virtual void SetUp() {
        model = nullptr;
    }

    virtual void TearDown() {
        if (model != nullptr) delete model;
    }
};

TEST_F(SpatialConvolutionTest, NAME_CHECK) {
    SpatialConvolution<double> spatialConvolution(0, 0, 0, 0, 0, 0, 0, 0);
    ASSERT_STREQ("SpatialConvolution", spatialConvolution.name);
}

TEST_F(SpatialConvolutionTest, FORWARD_in1out1kW2kH2dW1dH1padW0padH0_intputW3H3_outputW2H2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({1, 3, 3}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = 0.1;    input->data[1] = 0.2;    input->data[2] = 0.3;
    input->data[3] = 0.4;    input->data[4] = 0.5;    input->data[5] = 0.6;
    input->data[6] = 0.7;    input->data[7] = 0.8;    input->data[8] = 0.9;

    SpatialConvolution<double> spatialConvolution(1, 1, 2, 2, 1, 1, 0, 0);

    spatialConvolution.weight->data[0] = -0.0732;     spatialConvolution.weight->data[1] = 0.1398;
    spatialConvolution.weight->data[2] = -0.2506;     spatialConvolution.weight->data[3] = -0.4502;

    spatialConvolution.bias->data[0] = 0.01 * -3.0455;

    output = spatialConvolution.forward(input);
    ASSERT_NEAR(-0.335155, output->data[0], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in1out2kW2kH2dW1dH1padW0padH0_intputW3H3_outputW2H2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({1, 3, 3}));
    shared_ptr<Tensor<double>> output;

    input->data[0] = 0.1;    input->data[1] = 0.2;    input->data[2] = 0.3;
    input->data[3] = 0.4;    input->data[4] = 0.5;    input->data[5] = 0.6;
    input->data[6] = 0.7;    input->data[7] = 0.8;    input->data[8] = 0.9;

    AsciiModelParser asciiModelParser("data/model1.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(-0.16469337714370186, output->data[0], 1e-06);
    ASSERT_NEAR(-0.10020582221913871, output->data[1], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in3out5kW2kH2dW1dH1padW0padH0_intputW4H3_outputW3H2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({3, 3, 4}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.1;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.2;

    input->data[i++] = 0.9;    input->data[i++] = 0.8;    input->data[i++] = 0.7;    input->data[i++] = 0.1;
    input->data[i++] = 0.6;    input->data[i++] = 0.5;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.3;    input->data[i++] = 0.2;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.2;    input->data[i++] = 0.4;    input->data[i++] = 0.6;    input->data[i++] = 0.2;
    input->data[i++] = 0.1;    input->data[i++] = 0.3;    input->data[i++] = 0.5;    input->data[i++] = 0.1;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.7;    input->data[i++] = 0.5;

    AsciiModelParser asciiModelParser("data/model2.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(-0.77962702055925703, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.78742143599060599, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.28374509060528746, output->data[4 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.1558558491830872, output->data[4 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 2], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in5out3kW2kH2dW1dH1padW0padH0_intputW5H3_outputW4H2) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;


    AsciiModelParser asciiModelParser("data/model3.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(-0.097078148906743655, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.43489173512378609, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.65123419444634634, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 3], 1e-06);
    ASSERT_NEAR(0.77362594732718248, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.076218431533111586, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.32206086406090917, output->data[2 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 3], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in5out3kW3kH3dW1dH1padW0padH0_intputW5H3_outputW3H1) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;


    AsciiModelParser asciiModelParser("data/model4.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(-0.33405540649467291, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.41174322072731651, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.63579310876717399, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.12742073715436616, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.11037514674556717, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.024313361536482003, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.23942693657957234, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.21564188937174983, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.029410807003286771, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 2], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in5out4kW2kH2dW2dH2padW0padH0_intputW5H3_outputW2H1) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;


    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;


    AsciiModelParser asciiModelParser("data/model5.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(0.4623200669633673, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.14749530866120333, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.036379147453718091, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(-0.23214547043170092, output->data[3 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in1out1kW2kH2dW1dH1padW1padH1_intputW3H3_outputW4H4) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({1, 3, 3}));
    shared_ptr<Tensor<double>> output;

    int i = 0;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;


    AsciiModelParser asciiModelParser("data/model6.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(0.38519542261492801, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.47903489645105102, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.55169703334104292, output->data[0 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(-0.016519187623633025, output->data[0 * output->sizes[1] * output->sizes[2] + 3 * output->sizes[2] + 3], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_in5out4kW2kH2dW2dH2padW2padH2_intputW5H3_outputW4H3) {
    shared_ptr<Tensor<double>> input = make_shared<Tensor<double>>(vector<long>({5, 3, 5}));
    shared_ptr<Tensor<double>> output;

    int i = 0;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;

    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;
    
    input->data[i++] = 0.1;    input->data[i++] = 0.2;    input->data[i++] = 0.3;    input->data[i++] = 0.4;    input->data[i++] = 0.5;
    input->data[i++] = 0.4;    input->data[i++] = 0.5;    input->data[i++] = 0.6;    input->data[i++] = 0.7;    input->data[i++] = 0.8;
    input->data[i++] = 0.7;    input->data[i++] = 0.8;    input->data[i++] = 0.9;    input->data[i++] = 0.1;    input->data[i++] = 0.2;


    AsciiModelParser asciiModelParser("data/model7.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_NEAR(0.030239393745528999, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.030239393745528999, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.030239393745528999, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.030239393745528999, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 3], 1e-06);
    ASSERT_NEAR(0.030239393745528999, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.048374429369550505, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.33443001472461642, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.15716681836642432, output->data[2 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(-0.13193757142017595, output->data[3 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 3], 1e-06);
}

TEST_F(SpatialConvolutionTest, FORWARD_MM_in5out4kW2kH2dW2dH2padW2padH2_intputW5H3_outputW4H3) {
    shared_ptr<Cube<double>> input = make_shared<Cube<double>>(5, 3, 5);
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


    AsciiModelParser asciiModelParser("data/model12.tpa");
    ASSERT_NO_THROW(model = asciiModelParser.parse<double>());

    SpatialConvolution<double>* spatialConvolution = (SpatialConvolution<double> *) model;
    ASSERT_TRUE(spatialConvolution != nullptr);

    output = spatialConvolution->forward(input);
    ASSERT_EQ(4, output->sizes[0]);
    ASSERT_NEAR(0.13314539279237, output->data[0 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.51515939833061719, output->data[0 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.1624306994215492, output->data[0 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.11835824783503, output->data[1 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 3], 1e-06);
    ASSERT_NEAR(0.11835824783503, output->data[1 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.49019795572222086, output->data[1 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(-0.082984111946570002, output->data[2 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.16983744113412919, output->data[2 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 3], 1e-06);
    ASSERT_NEAR(-0.082984111946570002, output->data[2 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 0], 1e-06);
    ASSERT_NEAR(0.13368789367035999, output->data[3 * output->sizes[1] * output->sizes[2] + 0 * output->sizes[2] + 1], 1e-06);
    ASSERT_NEAR(0.16032582178384114, output->data[3 * output->sizes[1] * output->sizes[2] + 1 * output->sizes[2] + 2], 1e-06);
    ASSERT_NEAR(0.12276702088167563, output->data[3 * output->sizes[1] * output->sizes[2] + 2 * output->sizes[2] + 3], 1e-06);
}
