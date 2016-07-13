require 'torch'
require 'nn'

convolutionModule = nn.SpatialConvolution(3, 5, 2, 2, 1, 1, 0, 0)

model = nn.Sequential()
model:add(convolutionModule)

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

torch.save('3-5.t7', model)

print('w')
print(convolutionModule.weight)
print('b')
print(convolutionModule.bias)
print('x')
print(x)
print('y')
print(y)