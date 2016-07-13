require 'torch'
require 'nn'

model = nn.Sequential()
model:add(nn.View(5*3*5))
model:add(nn.Dropout(0))
model:add(nn.Threshold())
model:add(nn.Linear(5*3*5, 50))

x = torch.Tensor(5, 3, 5)
-- x[1]
x[1][1][1] = 0.1; 	x[1][1][2] = 0.2; 	x[1][1][3] = 0.3;	x[1][1][4] = 0.4 	x[1][1][5] = 0.5
x[1][2][1] = 0.4;	x[1][2][2] = 0.5; 	x[1][2][3] = 0.6;	x[1][2][4] = 0.7 	x[1][2][5] = 0.8
x[1][3][1] = 0.7; 	x[1][3][2] = 0.8; 	x[1][3][3] = 0.9;	x[1][3][4] = 0.1 	x[1][3][5] = 0.2
-- x[2]
x[2][1][1] = -0.1; 	x[2][1][2] = -0.2; 	x[2][1][3] = -0.3;	x[2][1][4] = -0.4	x[2][1][5] = -0.5
x[2][2][1] = -0.4;	x[2][2][2] = -0.5; 	x[2][2][3] = -0.6;	x[2][2][4] = -0.7	x[2][2][5] = -0.8
x[2][3][1] = -0.7; 	x[2][3][2] = -0.8; 	x[2][3][3] = -0.9;	x[2][3][4] = -0.1	x[2][3][5] = -0.2
-- x[3]
x[3][1][1] = 0.1; 	x[3][1][2] = 0.2; 	x[3][1][3] = 0.3;	x[3][1][4] = 0.4	x[3][1][5] = 0.5
x[3][2][1] = 0.4;	x[3][2][2] = 0.5; 	x[3][2][3] = 0.6;	x[3][2][4] = 0.7	x[3][2][5] = 0.8
x[3][3][1] = 0.7; 	x[3][3][2] = 0.8; 	x[3][3][3] = 0.9;	x[3][3][4] = 0.1	x[3][3][5] = 0.2
-- x[4]
x[4][1][1] = 0.1; 	x[4][1][2] = 0.2; 	x[4][1][3] = 0.3;	x[4][1][4] = 0.4	x[4][1][5] = 0.5
x[4][2][1] = 0.4;	x[4][2][2] = 0.5; 	x[4][2][3] = 0.6;	x[4][2][4] = 0.7	x[4][2][5] = 0.8
x[4][3][1] = 0.7; 	x[4][3][2] = 0.8; 	x[4][3][3] = 0.9;	x[4][3][4] = 0.1	x[4][3][5] = 0.2
-- x[5]
x[5][1][1] = -0.1; 	x[5][1][2] = 0.2; 	x[5][1][3] = -0.3;	x[5][1][4] = 0.4	x[5][1][5] = -0.5
x[5][2][1] = 0.4;	x[5][2][2] = -0.5; 	x[5][2][3] = 0.6;	x[5][2][4] = -0.7	x[5][2][5] = 0.8
x[5][3][1] = -0.7; 	x[5][3][2] = 0.8; 	x[5][3][3] = -0.9;	x[5][3][4] = 0.1	x[5][3][5] = -0.2


y = model:forward(x)

torch.save('model19.t7', model)

print('y')
print(y)
