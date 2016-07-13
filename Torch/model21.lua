require 'torch'
require 'nn'
require 'image'

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

x = image.load('lena.jpg')
print(x)

y = model:forward(x)

torch.save('model21.t7', model)

print('y')
print(y)
