require 'torch'
require 'nn'

maxPoolingModule = nn.SpatialMaxPooling(1, 1, 1, 1, 0, 0)
model = nn.Sequential()
model:add(maxPoolingModule)

x = torch.Tensor(1, 3, 3)
-- x[1]
x[1][1][1] = 0.1; 	x[1][1][2] = 0.2; 	x[1][1][3] = 0.3;
x[1][2][1] = 0.4;	x[1][2][2] = 0.5; 	x[1][2][3] = 0.6;
x[1][3][1] = 0.7; 	x[1][3][2] = 0.8; 	x[1][3][3] = 0.9;


y = model:forward(x)

torch.save('model8.t7', model)
print('y')
print(y)
