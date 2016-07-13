require 'torch'
require 'nn'

model = nn.SoftMax()

x = torch.Tensor(2, 3)
-- x[1]
x[1][1] = 0.1;	x[1][2] = 0.2;	x[1][3] = -0.3;
x[2][1] = 0.4;	x[2][2] = 0.5;	x[2][3] = 0.6;


y = model:forward(x)

torch.save('model13.t7', model)

print('y')
print(y)
