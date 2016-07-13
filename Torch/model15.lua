require 'torch'
require 'nn'

model = nn.View(2 * 1 * 3)

x = torch.Tensor(2, 1, 3)
-- x[1]
x[1][1][1] = 0.1; 	x[1][1][2] = 0.2; 	x[1][1][3] = 0.3;
-- x[2]
x[2][1][1] = -0.1; 	x[2][1][2] = -0.2; 	x[2][1][3] = -0.3;


y = model:forward(x)

torch.save('model15.t7', model)

print('y')
print(y)
