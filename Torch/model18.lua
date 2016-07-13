require 'torch'
require 'nn'

model = nn.Linear(3, 2)

x = torch.Tensor(3)
x[1] = 0.1; x[2] = 0.2; x[3] = 0.3;


y = model:forward(x)

torch.save('model18.t7', model)

print('y')
print(y)
