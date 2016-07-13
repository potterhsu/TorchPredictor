require 'torch'
require 'nn'

convolutionModule = nn.SpatialConvolution(1, 1, 2, 2, 1, 1, 1, 1)
model = nn.Sequential()
model:add(convolutionModule)

x = torch.Tensor(1, 3, 3)
-- x[1]
x[1][1][1] = 0.1; 	x[1][1][2] = 0.2; 	x[1][1][3] = 0.3;
x[1][2][1] = 0.4;	x[1][2][2] = 0.5; 	x[1][2][3] = 0.6;
x[1][3][1] = 0.7; 	x[1][3][2] = 0.8; 	x[1][3][3] = 0.9;


y = model:forward(x)

torch.save('model6.t7', model)

print('w')
print(convolutionModule.weight)
print('b')
print(convolutionModule.bias)
print('x')
print(x)
print('y')
print(y)
