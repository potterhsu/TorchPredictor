require 'torch'
require 'nn'
require 'image'

model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())

x = image.load('lena.jpg')
print(x)

y = model:forward(x)

torch.save('model23.t7', model)

print('y')
print(y)
