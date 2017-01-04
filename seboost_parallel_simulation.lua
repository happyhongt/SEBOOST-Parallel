
local function copy2(obj)
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[copy2(k)] = copy2(v) end
  return res
end

--[[ A implementation of seboost

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX.
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.optMethod`         : The base optimizaion method
- `config.momentum`          : weight for SEBOOST's momentum direction
- `config.histSize`          : number of previous directions to keep in memory
- `config.anchorPoints`      : A tensor of values, each describing the number of             iterations between an update of an anchor point
- `config.sesopUpdate`       : The number of regular optimization steps between each boosting step
- `config.sesopData`         : The training data to use for the boosting stage
- `config.sesopLabels`       : The labels to use for the boosting stage
- `config.sesopBatchSize`    : The number of samples to use for each optimization step
- `config.isCuda`            : Whether to train using cuda or cpu
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.itr`               : The current optimization iteration
- `state.dirs`              : The set of directions to optimize in
- `state.anchors`           : The current anchor points
- `state.aOpt`              : The current set of optimal coefficients
- `state.dirIdx`            : The next direction to override

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

require 'optim'


--[[
SV:
  We have 'config.numNodes' nodes.
  All nodes start from the same 'x'.
  Every node does 'state.nodeIters' baseMethod iterations.
  
  Once all the nodes are done, we do a sesop iteration to "merge"
  The states of the different nodes.
  
  On true parallel execution, the nodes will execute in parallel, 
  so the time for one external iteration will be:
  T_p = T(baseMethod)*state.nodeIters + T(sesop)
  
  On our simulation, the time is:
  T_s = T(baseMethod)*state.nodeIters*config.numNodes + T(sesop)
  
  Assume there are k external iteration in an epoch. 
  So assume number of internal iterations in an epoch divides state.nodeIters.
  I.e., state.nodeIters*k = number of iterations in an epoch:
  
  T_s(epoch) = (T(baseMethod)*state.nodeIters*config.numNodes + T(sesop))*k = 
    T_s(baseMethod)*(number of iterations in an epoch)*config.numNodes + T(sesop)*k
    
  T_p(epoch) = (T(baseMethod)*state.nodeIters + T(sesop))*k = 
    T(baseMethod)*(number of iterations in an epoch) + T(sesop)*k
    
  In our first expr, we assume T(sesop) is neglible, and we compute the expected parallel time as:
  
  T_p(epoch) = T_s(epoch)/config.numNodes
  
  And we plot the graphs, we should see an improvemnt as we increase the number of nodes.
]]

function optim.seboost_tao(opfunc, x, config, state)

  -- get/update state
  local config = config or {}
  local state = state or config
  local isCuda = config.isCuda or true
  local sesopData = config.sesopData
  local sesopLabels = config.sesopLabels
  local sesopBatchSize = config.sesopBatchSize or 100
  -- Tao codes
  local state.histSize = state.histSize or 0
  if state.histSize ~=0 then
     local state.histspace = state.histspace or torch.zeros(x:size(1),state.histSize)
  end

  state.itr = state.itr or 0
  

  --number of iterations per node
  config.nodeIters = config.nodeIters or 100
	config.numNodes = config.numNodes or 2
  
	state.currNode = state.currNode or 0 --start from node 0
	state.lastNodeXs = state.lastNodeXs or {}
	state.splitPoint = state.splitPoint or x:clone() --the first split point is the first point
  state.sesopIteration = state.sesopIteration or 0
  

	local isMergeIter = false
  state.itr = state.itr + 1

	--node switch
  if (config.numNodes > 1 and state.itr % (config.nodeIters + 1) == 0) then
    --print ('In node switch '.. state.itr)
		--a node has finished. Save its last x location
		state.lastNodeXs[state.currNode] = x:clone()

		--progress to next node
		state.currNode = (state.currNode + 1)%config.numNodes

		--merge iteration (run seboost to merge).
		if (state.currNode == 0) then
			isMergeIter = true
		end

		--The new node starts from the split point
    x:copy(state.splitPoint)
  end
  

  if (isMergeIter == false or config.numNodes == 1) then
    config.optConfig[state.currNode] = config.optConfig[state.currNode] or copy2(config.initState)
		x,fx = config.optMethod(opfunc, x, config.optConfig[state.currNode])
		return x,fx
	end

  --Now x is the split point.
  ------------------------- SESOP Part ----------------------------
  --print ('****************SESOP***********')
  --print ('--------------------------------')
  state.dirs = state.dirs or torch.zeros(x:size(1), config.numNodes)  
  --SV, build directions matrix
  for i = 0, config.numNodes - 1 do   
    --[{ {}, i }] means: all of the first dim, slice in the second dim at i = get i col.
    state.dirs[{ {}, i + 1 }]:copy(state.lastNodeXs[i] - state.splitPoint) 
  end

--Tao Code
  local temp_dir
  if state.histSize~=0 then
     temp_dir = torch.cat(state.dirs,state.histspace,2)
  else 
     temp_dir = state.dirs
  end

  state.aOpt = torch.ones(temp_dir:size(2))*(1/temp_dir:size(2)) --avrage
  
  if isCuda then
    temp_dir = temp_dir:cuda()
    state.aOpt = state.aOpt:cuda()
  end
  

  --now optimize!
  local xInit = state.splitPoint
    -- create mini batch
  local subT = (state.sesopIteration) * sesopBatchSize + 1
  subT = subT % (sesopData:size(1) - sesopBatchSize) --Calculate the next batch index
  local sesopInputs = sesopData:narrow(1, subT, sesopBatchSize)
  local sesopTargets = sesopLabels:narrow(1, subT, sesopBatchSize)

  if isCuda then
     sesopInputs = sesopInputs:cuda()
     sesopTargets = sesopTargets:cuda()
  end

  -- Create inner opfunc for finding a*
  local feval = function(a)
    --A function of the coefficients
    local dirMat = temp_dir
    --Note that opfunc also gets the batch
    local afx, adfdx = opfunc(xInit + dirMat*a, sesopInputs, sesopTargets)
    return afx, (dirMat:t()*adfdx)
  end

  --x,f(x)
  config.maxIter = config.numNodes
  local _, fHist = optim.cg(feval, state.aOpt, config, state) --Apply optimization using inner function
   
  --updating model weights!
  x:copy(xInit)
  local sesopDir = state.dirs*state.aOpt 
  x:add(sesopDir)
  

  --Tao code update the history direction here
  if state.histSize~=0 then
    state.histspace = torch.cat(x - xInit,state.histspace:narrow(2,1,state.histSize-1),1)
  end
 
  --the new split point is 'x'.
  --The next time this function is called will be with 'x'.
  --The next time we will change a node, it will get this 'x'.
  state.splitPoint:copy(x)
  state.sesopIteration = state.sesopIteration + 1
  return x,fHist
  
end

return optim

