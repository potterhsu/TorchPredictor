require 'torch'
require 'nn'

---
-- Load Torch model, and print attribute and parameter of Layer for reading by other Applications.
-- For example, with respect to SpatialConvolution layer:
-- 1. Attribute
--		(1) nInputPlane
--		(2) nOutputPlane
--		(3) kW
--		(4) kH
--		(5) dW
--		(6) dH
--		(7) padW
--		(8) padH
-- 2. Parameter
--		(1) Wieght
--		(2) Bias

local pathToModel = arg[1]
local model = torch.load(pathToModel)

for i, module in ipairs(model:listModules()) do
	local moduleName = torch.type(module)
	if moduleName == 'nn.Sequential' then
		print(moduleName)
	elseif moduleName == 'nn.SpatialConvolution' or moduleName == 'nn.SpatialConvolutionMM' then
		print('nn.SpatialConvolution')
		print('nInputPlane'); print(module.nInputPlane)
		print('nOutputPlane'); print(module.nOutputPlane)
		print('kW'); print(module.kW)
		print('kH'); print(module.kH)
		print('dW'); print(module.dW)
		print('dH'); print(module.dH)
		print('padW'); print(module.padW)
		print('padH'); print(module.padH)

		print('weight')
		local weight = module.weight
		if moduleName == 'nn.SpatialConvolutionMM' then
			weight = torch.reshape(weight, module.nOutputPlane, module.nInputPlane, module.kW, module.kH)
		end
		local weightSizes = weight:size()
		for i = 1, weightSizes[1] do
			for j = 1, weightSizes[2] do
				for r = 1, weightSizes[3] do
					for c = 1, weightSizes[4] do
						print(weight[i][j][r][c])
					end
				end
			end	
		end

		print('bias')
		local biasSize = module.bias:size()[1]
		for i = 1, biasSize do
			print(module.bias[i])
		end
	elseif moduleName == 'nn.SpatialMaxPooling' then
		print(moduleName)
		print('kW'); print(module.kW)
		print('kH'); print(module.kH)
		print('dW'); print(module.dW)
		print('dH'); print(module.dH)
		print('padW'); print(module.padW)
		print('padH'); print(module.padH)
	elseif moduleName == 'nn.ReLU' then
		print(moduleName)
	elseif moduleName == 'nn.SoftMax' then
		print(moduleName)
	elseif moduleName == 'nn.LogSoftMax' then
		print(moduleName)
	elseif moduleName == 'nn.View' then
		print(moduleName)
		print('size'); print(module.size[1])
	elseif moduleName == 'nn.Dropout' then
		print(moduleName)
		print('p'); print(module.p)
	elseif moduleName == 'nn.Threshold' then
		print(moduleName)
		print('threshold'); print(string.format("%.8f", module.threshold))
	elseif moduleName == 'nn.Linear' then
		print(moduleName)
		local weightSizes = module.weight:size()
		local inputSize = weightSizes[2]
		local outputSize = weightSizes[1]
		print('inputSize'); print(inputSize)
		print('outputSize'); print(outputSize)
		print('weight')
		for i = 1, outputSize do
			for j = 1, inputSize do
				print(module.weight[i][j])
			end	
		end
		print('bias')
		local biasSize = module.bias:size()[1]
		for i = 1, biasSize do
			print(module.bias[i])
		end
	end
end