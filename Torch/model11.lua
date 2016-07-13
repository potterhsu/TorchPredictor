require 'torch'
require 'nn'

reLUModule = nn.ReLU()
model = nn.Sequential()
model:add(reLUModule)

x = torch.Tensor(2, 2, 2)
-- x[1]
x[1][1][1] = 0.1; 	x[1][1][2] = 0.2;
x[1][2][1] = 0.4;	x[1][2][2] = 0.5;

-- x[2]
x[2][1][1] = -0.1; 	x[2][1][2] = 0.2;
x[2][2][1] = 0.4;	x[2][2][2] = -0.5;


y = model:forward(x)

torch.save('model11.t7', model)
print('y')
print(y)
