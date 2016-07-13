# Model Description #

## model1 ##
    convolutionModule = nn.SpatialConvolution(1, 2, 2, 2, 1, 1, 0, 0)
    model = nn.Sequential()
    model:add(convolutionModule)

## model2 ##
    convolutionModule = nn.SpatialConvolution(3, 5, 2, 2, 1, 1, 0, 0)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model3 ##
    convolutionModule = nn.SpatialConvolution(5, 3, 2, 2, 1, 1, 0, 0)
    model = nn.Sequential()
    model:add(convolutionModule)

## model4 ##
    convolutionModule = nn.SpatialConvolution(5, 3, 3, 3, 1, 1, 0, 0)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model5 ##
    convolutionModule = nn.SpatialConvolution(5, 4, 2, 2, 2, 2, 0, 0)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model6 ##
    convolutionModule = nn.SpatialConvolution(1, 1, 2, 2, 1, 1, 1, 1)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model7 ##
    convolutionModule = nn.SpatialConvolution(5, 4, 2, 2, 2, 2, 2, 2)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model8 ##
    maxPoolingModule = nn.SpatialMaxPooling(1, 1, 1, 1, 0, 0)
    model = nn.Sequential()
    model:add(maxPoolingModule)
    
## model9 ##
    maxPoolingModule = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1)
    model = nn.Sequential()
    model:add(maxPoolingModule)
    
## model10 ##
    convolutionModule = nn.SpatialConvolution(5, 5, 3, 3, 2, 2, 1, 1)
    maxPoolingModule = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1)
    model = nn.Sequential()
    model:add(convolutionModule)
    model:add(maxPoolingModule)
    
## model11 ##
    reLUModule = nn.ReLU()
    model = nn.Sequential()
    model:add(reLUModule)
    
## model12 ##
*model12.tpa* is convert model which includes **nn.SpatialConvolutionMM**,
however, after **ModelPrinter** the module name has been normalized to **nn.SpatialConvolution**

    convolutionModule = nn.SpatialConvolutionMM(5, 4, 2, 2, 2, 2, 2, 2)
    model = nn.Sequential()
    model:add(convolutionModule)
    
## model13 ##

    model = nn.SoftMax()
    
## model14 ##

    model = nn.LogSoftMax()
    
## model15 ##

    model = nn.View(2 * 1 * 3)
    
## model16 ##
Dropout has a parameter **p** which default is 0.5

    model = nn.Dropout()
    
## model17 ##
Threshold has a parameter **threshold** which default is 0.000001

    model = nn.Threshold()
    
## model18 ##

    model = nn.Linear(3, 2)

## model19 ##

    model = nn.Sequential()
    model:add(nn.View(5*3*5))
    model:add(nn.Dropout())
    model:add(nn.Threshold())
    model:add(nn.Linear(5*3*5, 50))

## model20 ##

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(5, 4, 2, 2, 2, 2, 2, 2))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
    model:add(nn.ReLU())
    model:add(nn.View(4*2*2))
    model:add(nn.Dropout(0))
    model:add(nn.Threshold())
    model:add(nn.Linear(4*2*2, 100))
    model:add(nn.LogSoftMax())

## model21 ##

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3, 16, 11, 11, 4, 4, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    model:add(nn.SpatialConvolution(16, 6, 5, 5, 1, 1, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    model:add(nn.SpatialConvolution(6, 12, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(12, 6, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    model:add(nn.Sequential())
    model:add(nn.View(6*7*7))
    model:add(nn.Dropout(0))
    model:add(nn.Threshold())
    model:add(nn.Linear(6*7*7, 4))
    model:add(nn.LogSoftMax())

## model22 ##

    model = nn.View(-1)

## model23 ##

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())
    
    
***


# Flow for Adding New Module #

For example, assume you want to add **SoftMax** module:

### Prepare SoftMax module class and its test class ###

1. Under *CPP/lib/nn/*, add **SoftMax** class
2. Let **SoftMax** extend **Module**, then generate constructor and implement **forward** method (remove **virtual** syntax if exists)
3. Under *CPP/test*, add *.cpp* and *.h* of **SoftMaxTest**
4. Add inclusion of *gtest.h* in header file
5. Add inclusion of *SoftMax.h* in source file, and add first test case:
#
    TEST(SOFTMAX, NAME_CHECK) {
        SoftMax softMax;
        ASSERT_STREQ("SoftMax", softMax.name);
    }
    
6. Edit *TestAll.cpp*, and add inclusion of *SoftMaxTest.h*, thus it can be tested to
7. Now, run **TestAll**, you should get an error message about asserting failed
8. Back to *SoftMax.cpp*, setup **name** and **type** for **SoftMax**
9. Run **TestAll** again, this time should be passed
10. Add other test cases for **forward** method, you can also use **AsciiModelParser** to help test 

### Extend AsciiModelParser ###

1. Under *Torch/*, add model#.lua from copy of *modelX.lua*, modify model and output filename of *model#.t7*
2. Edit *1_generate-t7.sh* and *2_convert-ty-to-tpa.sh*, change their **MODEL** variable
3. Under *CPP/test* Edit *ModelPrinter.lua*, add condition branch for new module
4. Run above two bash scripts, it will generate *model#.t7* and *model#.tpa*
5. Copy *model#.tpa* to *CPP/test/data/* and add description to above section in this file
6. Modify **AsciiModelParserTest**, **AsciiModelParser** in sequence, and test it until pass


***


# Sample for Loading Model from Torch7 #
#### You can use following code of Torch to predict  ####

    require 'torch'
    require 'nn'
    
    model = torch.load('model2.t7')
    
    x = torch.Tensor(3, 3, 4)
    -- x[1]
    x[1][1][1] = 0.1; 	x[1][1][2] = 0.2; 	x[1][1][3] = 0.3;	x[1][1][4] = 0.5
    x[1][2][1] = 0.4;	x[1][2][2] = 0.5; 	x[1][2][3] = 0.6;	x[1][2][4] = 0.1
    x[1][3][1] = 0.7; 	x[1][3][2] = 0.8; 	x[1][3][3] = 0.9;	x[1][3][4] = 0.2
    -- x[2]
    x[2][1][1] = 0.9; 	x[2][1][2] = 0.8; 	x[2][1][3] = 0.7;	x[2][1][4] = 0.1
    x[2][2][1] = 0.6;	x[2][2][2] = 0.5; 	x[2][2][3] = 0.4;	x[2][2][4] = 0.5
    x[2][3][1] = 0.3; 	x[2][3][2] = 0.2; 	x[2][3][3] = 0.1;	x[2][3][4] = 0.2
    -- x[3]
    x[3][1][1] = 0.2; 	x[3][1][2] = 0.4; 	x[3][1][3] = 0.6;	x[3][1][4] = 0.2
    x[3][2][1] = 0.1;	x[3][2][2] = 0.3; 	x[3][2][3] = 0.5;	x[3][2][4] = 0.1
    x[3][3][1] = 0.7; 	x[3][3][2] = 0.8; 	x[3][3][3] = 0.7;	x[3][3][4] = 0.5
    
    y = model:forward(x)
    print(y)