require 'cunn'
require 'cutorch'
cutorch.setDevice(1)
require 'image'
require 'gnuplot'
local trainset = torch.load('/home/tao/Project_torch/Double Sparsity CNN/cifar10-train.t7')
local testset = torch.load('/home/tao/Project_torch/Double Sparsity CNN/cifar10-test.t7')

local isCUDA = true

local classes = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'}

function shuffle(data,ydata)
	local RandOrder = torch.randperm(data:size(1)):long()
	return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

local trainData = trainset.data:double()
local trainLabels = trainset.label:double():add(1)

local testData = testset.data:double()
local testLabels = testset.label:double():add(1)
--image.display(trainData[100])
--print(classes[trainLabels[100]])

local mean = {}
local stdv = {}
for i=1,3 do
    mean[i] = trainData[{{},{i},{},{}}]:mean()
    trainData[{{},{i},{},{}}]:add(-mean[i])

    stdv[i] = trainData[{{},{i},{},{}}]:std()
    trainData[{{},{i},{},{}}]:div(stdv[i])
end

for i=1,3 do
    testData[{{},{i},{},{}}]:add(-mean[i])
    testData[{{},{i},{},{}}]:div(stdv[i])
end

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,32,5,5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialBatchNormalization(32))
model:add(nn.SpatialConvolution(32,64,3,3))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.SpatialConvolution(64,32,3,3))
model:add(nn.View(32*4*4):setNumInputDims(3))
model:add(nn.Linear(32*4*4,256))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(256,#classes))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

if isCUDA then
   model,criterion = model:cuda(),criterion:cuda()
end

local w,dE_dw = model:getParameters()

print(w:nElement())

require 'optim'

local batchSize = 128
local optimState = {
	
}

function forwardNet(data,labels,train)
	local confusion = optim.ConfusionMatrix(classes)
	local lossAcc = 0
	local numBatches = 0
	if train then
		model:training()
	else
		model:evaluate()	
	end

    for i=1,data:size(1)-batchSize,batchSize do
    	numBatches = numBatches+1
        local x = data:narrow(1,i,batchSize)
        local yt = labels:narrow(1,i,batchSize)
        if isCUDA then
        	x,yt = x:cuda(),yt:cuda()
        end
        local y = model:forward(x)
        local err = criterion:forward(y,yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)

        if train then
        	function feval()
        		model:zeroGradParameters()
        		local dE_dy = criterion:backward(y,yt)
        		model:backward(x,dE_dy)
        		return err,dE_dw
        	end

          optim.adam(feval,w,optimState)
        end
    end
        confusion:updateValids()
   
   local avgLoss = lossAcc/numBatches
   local avgError = 1-confusion.totalValid

   return avgLoss,avgError,tostring(confusion)
end

function plotError(trainError,testError,title)
	local range = torch.range(1,trainError:size(1))
    
    gnuplot.plot({'Training error',range,trainError*100,'-'},{'Test error',range,testError*100,'-'})
    gnuplot.title(title)
end

local epochs = 50
local trainLoss = torch.Tensor(epochs)
local testLoss = torch.Tensor(epochs)
local trainError = torch.Tensor(epochs)
local testError = torch.Tensor(epochs)

model:apply(function(l) l:reset() end)

logger_loss = optim.Logger('loss.log')

logger_error = optim.Logger('error.log')

for e = 1,epochs do
	trainData,trainLabels = shuffle(trainData,trainLabels)
	trainLoss[e],trainError[e]= forwardNet(trainData,trainLabels,true)
    testLoss[e],testError[e],confusion = forwardNet(testData,testLabels,false)
    logger_loss:add{['training loss'] = trainLoss[e],['test loss'] = testLoss[e]}
    logger_error:add{['training loss'] = trainError[e],['test loss'] = testError[e]}
    logger_loss:plot()
    logger_error:plot()
    if e%5 ==0 then
    	--print('Epoch' .. e .. ';')
    	--print('Training error:' .. trainError[e],'Training Loss:' .. trainLoss[e])
    	--print('Test error:' .. testError[e],'Test Loss:' .. testLoss[e])
    	print(confusion)
    end
end

--plotError(trainError,testError,'Classification Error')

torch.save('cifar10_model.t7',model)
