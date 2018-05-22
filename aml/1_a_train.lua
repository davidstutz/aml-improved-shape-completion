-- Train an encoder using config.json.

require('torch')
require('nn')
require('optim')
require('hdf5')
require('cunn')
require('lfs')

package.path = package.path .. ';../?/th/init.lua'
lib = require('lib')

-- https://github.com/torch/cutorch
if cutorch.isCachingAllocatorEnabled() then
  print('[Training] caching allocator enabled')
end

--- Convert a table of tensors to a table of CUDA tensors.
-- @param table the table to convert
local function tableToCUDA(table)
  for key, value in pairs(table) do
    table[key] = table[key]:cuda()
  end
end

--- Set all table elements to zero.
-- @param table table to set to zero
local function tableToZero(table)
  for key, value in pairs(table) do
    table[key] = 0
  end
end

--- Append the tensor tensor to the tensor acc which may initially be nil.
local function appendTensor(acc, tensor, dim)
  local dim = dim or 1
  if acc == nil then
    acc = tensor:float()
  else
    acc = torch.cat(acc, tensor:float(), dim)
  end

  return acc
end

-- Load configuration.
configFile = 'config.json'
if arg[1] and lib.utils.fileExists(arg[1]) then
  configFile = arg[1]
end

print('[Training] reading ' .. configFile)
config = lib.utils.readJSON(configFile)

config['vae_aml_parameters']['model_file'] = config['base_directory'] .. config['vae_aml_parameters']['model_file']
config['prior_parameters']['model_file'] = config['base_directory'] .. '/' .. config['prior_parameters']['model_file']
config['base_directory'] = config['base_directory'] .. 'vae_aml/'
config['test_directory'] = config['base_directory'] .. config['test_directory']
config['train_protocol_file'] = config['base_directory'] .. config['train_protocol_file']
config['val_protocol_file'] = config['base_directory'] .. config['val_protocol_file']
config['early_stop_file'] = config['base_directory'] .. config['early_stop_file']

keys = {
  'inference_training_inputs',
  'inference_training_sdf_inputs',
  'inference_training_outputs',
  'inference_training_sdf_outputs',
  'inference_training_space',
  'validation_inputs',
  'validation_sdf_inputs',
  'validation_outputs',
  'validation_sdf_outputs',
  'validation_space',
}

for i, key in pairs(keys) do
  config[key] = config['data_directory'] .. '/' .. config[key]
  if not lib.utils.fileOrDirectoryExists(config[key]) then
    print('[Error] file or directory ' .. config[key] .. ' does not exist')
    os.exit()
  end
end

if not lib.utils.fileExists(config['prior_parameters']['model_file']) then
  print('[Error] prior model file not found')
  os.exit()
end

-- Load data for validation.
print('[Training] reading ' .. config['inference_training_inputs'])
points = lib.utils.readHDF5(config['inference_training_inputs'])
nObservations = points:size(2)
points = points:resize(points:size(1)*points:size(2), 1, points:size(3), points:size(4), points:size(5))

print('[Training] reading ' .. config['inference_training_space'])
space = lib.utils.readHDF5(config['inference_training_space'])
space = space:resize(space:size(1)*space:size(2), 1, space:size(3), space:size(4), space:size(5))

--print('[Training] reading ' .. config['inference_training_inputs'])
--inputs_1 = lib.utils.readHDF5(config['inference_training_inputs'])
print('[Training] reading ' .. config['inference_training_sdf_inputs'])
inputs_2 = lib.utils.readHDF5(config['inference_training_sdf_inputs'])

--inputs_1 = inputs_1:resize(inputs_1:size(1)*inputs_1:size(2), 1, inputs_1:size(3), inputs_1:size(4), inputs_1:size(5))
inputs_2 = inputs_2:resize(inputs_2:size(1)*inputs_2:size(2), 1, inputs_2:size(3), inputs_2:size(4), inputs_2:size(5))
--inputs = torch.cat({inputs_1:float(), inputs_2:float()}, 2)
inputs = torch.cat({points:float(), inputs_2:float()}, 2)

print('[Training] reading ' .. config['inference_training_outputs'])
outputs_1 = lib.utils.readHDF5(config['inference_training_outputs'])
print('[Training] reading ' .. config['inference_training_sdf_outputs'])
outputs_2 = lib.utils.readHDF5(config['inference_training_sdf_outputs'])

outputs_1 = outputs_1:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*outputs_1:size(1), 1, outputs_1:size(3), outputs_1:size(4), outputs_1:size(5))
outputs_2 = outputs_2:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*outputs_2:size(1), 1, outputs_2:size(3), outputs_2:size(4), outputs_2:size(5))
outputs = torch.cat({outputs_1:float(), outputs_2:float()}, 2)

-- !
maxLTSDF = torch.max(outputs_2)
print('[Training] max LTSDF ' .. maxLTSDF)

print('[Training] reading ' .. config['validation_inputs'])
valPoints = lib.utils.readHDF5(config['validation_inputs'])
valPoints = valPoints:resize(valPoints:size(1)*valPoints:size(2), 1, valPoints:size(3), valPoints:size(4), valPoints:size(5))

print('[Training] reading ' .. config['validation_space'])
valSpace = lib.utils.readHDF5(config['validation_space'])
valSpace = valSpace:resize(valSpace:size(1)*valSpace:size(2), 1, valSpace:size(3), valSpace:size(4), valSpace:size(5))

--print('[Training] reading ' .. config['validation_inputs'])
--valInputs_1 = lib.utils.readHDF5(config['validation_inputs'])
print('[Training] reading ' .. config['validation_sdf_inputs'])
valInputs_2 = lib.utils.readHDF5(config['validation_sdf_inputs'])

--valInputs_1 = valInputs_1:resize(valInputs_1:size(1)*valInputs_1:size(2), 1, valInputs_1:size(3), valInputs_1:size(4), valInputs_1:size(5))
valInputs_2 = valInputs_2:resize(valInputs_2:size(1)*valInputs_2:size(2), 1, valInputs_2:size(3), valInputs_2:size(4), valInputs_2:size(5))
--valInputs = torch.cat({valInputs_1:float(), valInputs_2:float()}, 2)
valInputs = torch.cat({valPoints:float(), valInputs_2:float()}, 2)

print('[Training] reading ' .. config['validation_outputs'])
valOutputs_1 = lib.utils.readHDF5(config['validation_outputs'])
print('[Training] reading ' .. config['validation_sdf_outputs'])
valOutputs_2 = lib.utils.readHDF5(config['validation_sdf_outputs'])

valOutputs_1 = valOutputs_1:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*valOutputs_1:size(1), 1, valOutputs_1:size(3), valOutputs_1:size(4), valOutputs_1:size(5))
valOutputs_2 = valOutputs_2:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*valOutputs_2:size(1), 1, valOutputs_2:size(3), valOutputs_2:size(4), valOutputs_2:size(5))
valOutputs = torch.cat({valOutputs_1:float(), valOutputs_2:float()}, 2)

assert(points:size(1) == inputs:size(1))
assert(points:size(2) == 1)
assert(inputs:size(2) == 2)
assert(points:size(3) == inputs:size(3))
assert(points:size(4) == inputs:size(4))
assert(points:size(5) == inputs:size(5))

assert(points:size(1) == outputs:size(1))
assert(points:size(2) == 1)
assert(outputs:size(2) == 2)
assert(points:size(3) == outputs:size(3))
assert(points:size(4) == outputs:size(4))
assert(points:size(5) == outputs:size(5))

assert(points:size(1) == space:size(1))
assert(points:size(2) == space:size(2))
assert(points:size(3) == space:size(3))
assert(points:size(4) == space:size(4))
assert(points:size(5) == space:size(5))

assert(valPoints:size(1) == valInputs:size(1))
assert(valPoints:size(2) == 1)
assert(valInputs:size(2) == 2)
assert(valPoints:size(3) == valInputs:size(3))
assert(valPoints:size(4) == valInputs:size(4))
assert(valPoints:size(5) == valInputs:size(5))

assert(valPoints:size(1) == valOutputs:size(1))
assert(valPoints:size(2) == 1)
assert(valOutputs:size(2) == 2)
assert(valPoints:size(3) == valOutputs:size(3))
assert(valPoints:size(4) == valOutputs:size(4))
assert(valPoints:size(5) == valOutputs:size(5))

assert(valPoints:size(1) == valSpace:size(1))
assert(valPoints:size(2) == valSpace:size(2))
assert(valPoints:size(3) == valSpace:size(3))
assert(valPoints:size(4) == valSpace:size(4))
assert(valPoints:size(5) == valSpace:size(5))

-- Check dimensions.
N = inputs:size(1)
channels = inputs:size(2)
height = inputs:size(3)
width = inputs:size(4)
depth = inputs:size(5)

-- Create snapshot directory.
print('[Training] creating ' .. config['test_directory'])
if not lib.utils.directoryExists(config['test_directory']) then
  lib.utils.makeDirectory(config['test_directory'])
end

priorModel = torch.load(config['prior_parameters']['model_file'])
print(priorModel)

encoder = priorModel.modules[1]
mean = encoder.modules[#encoder.modules - 2].modules[1]
logvar = encoder.modules[#encoder.modules - 2].modules[2]
decoder = priorModel.modules[2]

KLD = encoder.modules[#encoder.modules - 1]
priorWeight = config['vae_aml_parameters']['prior_weight']
KLD.lambda = priorWeight
print('[Training] lambda ' .. KLD.lambda)

lib.utils.fixLayersAfter(decoder, 1)
if config['vae_aml_parameters']['reinitialize_encoder'] then
  lib.init(encoder, config['vae_aml_parameters']['weight_initialization'], config['vae_aml_parameters']['weight_value'], config['vae_aml_parameters']['bias_initialization'], config['vae_aml_parameters']['bias_value'])
  print('[Training] reinitializing encoder')
end

model = nn.Sequential()
model:add(encoder)
model:add(decoder)

model = model:cuda()
print(model)

-- Learning hyperparameters.
optimizer = optim.sgd
if config['vae_aml_parameters']['optimizer'] then
  if config['vae_aml_parameters']['optimizer'] == 'sgd' then
    print('[Training] using sgd')
  elseif config['vae_aml_parameters']['optimizer'] == 'adam' then
    optimizer = optim.adam
    print('[Training] using adam')
  elseif config['vae_aml_parameters']['optimizer'] == 'rmsprop' then
    optimizer = optim.rmsprop
    print('[Training] using rmsprop')
  else
    assert(false)
  end
end

batchSize = config['vae_aml_parameters']['batch_size']
learningRate = config['vae_aml_parameters']['learning_rate']
momentum = config['vae_aml_parameters']['momentum']
weightDecay = config['vae_aml_parameters']['weight_decay']
priorGoal = config['vae_aml_parameters']['prior_goal']
epochs = config['vae_aml_parameters']['epochs']
iterations = epochs*math.floor(N/batchSize)
minimumLearningRate = config['vae_aml_parameters']['minimum_learning_rate']
learningRateDecay = config['vae_aml_parameters']['decay_learning_rate']
maximumMomentum = config['vae_aml_parameters']['maximum_momentum']
momentumDecay = config['vae_aml_parameters']['decay_momentum']
decayIterations = config['vae_aml_parameters']['decay_iterations']
lossIterations = config['loss_iterations']
snapshotIterations = config['snapshot_iterations']
testIterations = config['test_iterations']
earlyStopError = 1e20

dataAugmentation = 0.5
if dataAugmentation > 0 then
  print('[Training] data augmentation')
end

if config['training_statistics'] then
  config['training_statistics'] = config['data_directory'] .. '/' .. config['training_statistics']
  print('[Training] reading ' .. config['training_statistics'])
  trainStatistics = lib.utils.readHDF5(config['training_statistics'])
  trainStatistics = nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(trainStatistics, 1), 1)
  trainStatistics = torch.repeatTensor(trainStatistics, batchSize, 1, 1, 1, 1)
end

sizeAverage = config['vae_aml_parameters']['size_average']
criterion = nn.MultipleCriterion()
criterion.weights = {}
criterion.criteria = {}
criterion.channels = {}

assert(#config['vae_aml_parameters']['criteria'] == #config['vae_aml_parameters']['weights'])
for i = 1, #config['vae_aml_parameters']['criteria'] do
  if config['vae_aml_parameters']['criteria'][i] == 'sdfpointbce' then
    local pointCriterion = nn.GaussianPointBCECriterion()
    pointCriterion.sizeAverage = false
    pointCriterion = pointCriterion:cuda()

    criterion.criteria[i] = pointCriterion
    criterion.weights[i] = config['vae_aml_parameters']['weights'][i]
    criterion.channels[i] = 2

  elseif config['vae_aml_parameters']['criteria'][i] == 'sdfspacebce' then
    local spaceCriterion = nn.GaussianFreeSpaceBCECriterion()
    spaceCriterion.sizeAverage = false
    spaceCriterion = spaceCriterion:cuda()

    if config['vae_aml_parameters']['weighted'] then
      assert(trainStatistics)

      spaceCriterion = nn.WeightedGaussianFreeSpaceBCECriterion()
      spaceCriterion.sizeAverage = sizeAverage
      spaceCriterion.weights = trainStatistics
      spaceCriterion = spaceCriterion:cuda()

      print('[Training] using weighted sdfspacebce')
    end

    criterion.criteria[i] = spaceCriterion
    criterion.weights[i] = config['vae_aml_parameters']['weights'][i]
    criterion.channels[i] = 2

  elseif config['vae_aml_parameters']['criteria'][i] == 'occpointbce' then
    local pointCriterion = nn.PointBCECriterion()
    pointCriterion.sizeAverage = false
    pointCriterion = pointCriterion:cuda()

    criterion.criteria[i] = pointCriterion
    criterion.weights[i] = config['vae_aml_parameters']['weights'][i]
    criterion.channels[i] = 1

  elseif config['vae_aml_parameters']['criteria'][i] == 'occspacebce' then
    local spaceCriterion = nn.FreeSpaceBCECriterion()
    spaceCriterion.sizeAverage = false
    spaceCriterion = spaceCriterion:cuda()

    if config['vae_aml_parameters']['weighted'] then
      assert(trainStatistics)

      spaceCriterion = nn.WeightedFreeSpaceBCECriterion()
      spaceCriterion.sizeAverage = sizeAverage
      spaceCriterion.weights = trainStatistics
      spaceCriterion = spaceCriterion:cuda()

      print('[Training] using weighted occspacebce')
    end

    criterion.criteria[i] = spaceCriterion
    criterion.weights[i] = config['vae_aml_parameters']['weights'][i]
    criterion.channels[i] = 1

  else
    assert(False)
  end
end

-- Parameters and gradients.
parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

-- Saves the loss.
protocol = torch.Tensor(iterations, 19):fill(0)
-- Will save: iteration, loss, KLD loss, error, thresh error, learning rate, momentum, lambda, occ loss, sdf loss, occ error, sdf error
valProtocol = torch.Tensor(math.floor(iterations/testIterations) + 1, 24):fill(0)
-- Wil save: iteration, loss x2, KLD loss x2, error x2, thresh error x2, mean mean, var mean, mean logvar

for t = 1, iterations do

  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()

  local container = {} -- Container will hold the different inputs and outputs in different representations.
  container['points'] = points:index(1, shuffle)
  container['space'] = space:index(1, shuffle)

  container['_input'] = inputs:index(1, shuffle)
  container['input'] = torch.Tensor(batchSize, 2, height, width, depth)
  container['_output'] = outputs:index(1, shuffle)
  container['output'] = torch.Tensor(batchSize, 2, height, width, depth)

  protocol[t][16] = 0
  if dataAugmentation then
    for b = 1, batchSize do
      local r = math.random()
      if r < dataAugmentation then
        local translate_h = torch.random(1, 4) - 2
        local translate_w = torch.random(1, 6) - 3
        local translate_d = torch.random(1, 4) - 2
        lib.translate_mirror(container['_output']:narrow(1, b, 1), container['output']:narrow(1, b, 1), translate_h, translate_w, translate_d)
        lib.translate_mirror(container['_input']:narrow(1, b, 1), container['input']:narrow(1, b, 1), translate_h, translate_w, translate_d)
        protocol[t][16] = protocol[t][16] + 1
      else
        container['output']:narrow(1, b, 1):copy(container['_output']:narrow(1, b, 1))
        container['input']:narrow(1, b, 1):copy(container['_input']:narrow(1, b, 1))
      end
    end
  else
    container['output'] = container['_output']
    container['input'] = container['_input']
  end

  tableToCUDA(container)

  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
  local feval = function(x)

    if x ~= parameters then
      parameters:copy(x)
    end

    gradParameters:zero()

    -- Evaluate function on mini-batch.
    local pred = model:forward(container['input'])
    local f = criterion:forward(pred, {container['points'], container['space'], container['points'], container['space']})
    local df = criterion:backward(pred, {container['points'], container['space'], container['points'], container['space']})
    model:backward(container['input'], df)

    -- Save losses in protocol.
    protocol[t][2] = f
    protocol[t][3] = KLD.loss

    protocol[t][17] = torch.mean(mean.output)
    protocol[t][18] = torch.std(mean.output)
    protocol[t][19] = torch.mean(logvar.output)

    -- Weight decay:
    if weightDecay > 0 then
      weightDecayLoss = weightDecay * torch.norm(parameters, 2)^2/2
      f = f + weightDecayLoss

      protocol[t][4] = weightDecayLoss
      gradParameters:add(parameters:clone():mul(weightDecay))
    end

    protocol[t][5] = torch.mean(torch.abs(pred - container['output']))
    protocol[t][12] = torch.mean(torch.abs(pred:narrow(2, 1, 1) - container['output']:narrow(2, 1, 1)))
    protocol[t][13] = torch.mean(torch.abs(pred:narrow(2, 2, 1) - container['output']:narrow(2, 2, 1)))

    local occPred = pred:narrow(2, 1, 1)
    occPred[occPred:gt(0.5)] = 1
    occPred[occPred:lt(0.5)] = 0
    protocol[t][14] = torch.mean(torch.abs(occPred - container['output']:narrow(2, 1, 1)))

    local sdfPred = pred:narrow(2, 2, 1)
    sdfPred[sdfPred:gt(0)] = 0
    sdfPred[sdfPred:lt(0)] = 1
    protocol[t][15] = torch.mean(torch.abs(sdfPred - container['output']:narrow(2, 1, 1)))

    -- Add the Kullback-Leibler loss.
    f = f + KLD.loss

    gradParameters:clamp(-1, 1)
    return f, gradParameters
  end

  -- Save learning rate and momentum in protocol.
  protocol[t][1] = t
  protocol[t][7] = learningRate
  protocol[t][8] = momentum
  protocol[t][9] = KLD.lambda

  -- Update state with learning rate and momentum.
  state = state or {
    learningRate = learningRate,
    momentum = momentum,
    learningRateDecay = 0 -- will be done manually below
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  --_, _ = optim.adam(feval, parameters, state)
  _, _ = optim.sgd(feval, parameters, state)
  local time = os.date("*t")

  -- Report a smoothed loss instead of batch loss.
  if t%lossIterations == 0 then

    -- Compute losses over the alst config['loss_iterations'] iterations.
    local smoothedLoss = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 2, 1))
    local smoothedKLD = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 3, 1))
    local smoothedWeightDecay = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 4, 1))

    if priorGoal > 0 then
      KLD.lambda = priorWeight*math.max(1, smoothedKLD/priorGoal)
    end

    local smoothedError = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 5, 1))
    local smoothedDataAugmentation = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 16, 1))

    local smoothedMean = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 17, 1))
    local smoothedStd = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 18, 1))
    local smoothedLogvar = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 19, 1))

    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ':'
            .. ' (' .. lib.utils.format_num(smoothedDataAugmentation) .. ')'
            .. ' [' .. lib.utils.format_num(smoothedLoss) .. ' | ' .. lib.utils.format_num(smoothedKLD) .. ']'
            .. ' [' .. lib.utils.format_num(smoothedMean) .. ' | ' .. lib.utils.format_num(smoothedStd) .. ' | ' .. lib.utils.format_num(smoothedLogvar) .. ']'
            .. ' [' .. lib.utils.format_num(smoothedError) .. ']')
  end

  -- Decay learning rate and KLD weight, do this before resetting all smoothed
  -- statistics.
  if t%decayIterations == 0 then
    learningRate = math.max(minimumLearningRate, learningRate*learningRateDecay)
    momentum = math.min(maximumMomentum, momentum*momentumDecay)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': learning rate ' .. learningRate)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': momentum ' .. momentum)
  end

  -- Validate on validation set.
  if t%testIterations == 0 or t == 1 or t == iterations then

    -- In case the validation set gets to large.
    local valN = valInputs:size(1)
    local valBatchSize = batchSize
    local valNumBatches = math.floor(valN/valBatchSize)
    local valIteration = math.floor(t/testIterations) + 1

    -- Accumulate and save all predictions.
    local accValPreds = nil
    local accValCodes = nil

    for b = 0, valNumBatches do
      local batchStart = b*valBatchSize + 1
      local batchLength = math.min((b + 1)*valBatchSize - b*valBatchSize, valN - b*valBatchSize)

      if batchLength == 0 then
        break;
      end

      local container = {} -- Container will hold the different inputs and outputs in different representations.
      container['input'] = valInputs:narrow(1, batchStart, batchLength)
      container['points'] = valPoints:narrow(1, batchStart, batchLength)
      container['space'] = valSpace:narrow(1, batchStart, batchLength)
      container['output'] = valOutputs:narrow(1, batchStart, batchLength)

      tableToCUDA(container)

      local valPreds = model:forward(container['input'])
      local valCodes = mean.output
      local f = criterion:forward(valPreds, {container['points'], container['space'], container['points'], container['space']})

      --accValPreds = appendTensor(accValPreds, valPreds)
      accValCodes = appendTensor(accValCodes, valCodes)

      valProtocol[valIteration][2] = valProtocol[valIteration][2] + f
      valProtocol[valIteration][4] = valProtocol[valIteration][4] + KLD.loss

      valProtocol[valIteration][6] = valProtocol[valIteration][6] + torch.mean(torch.abs(valPreds - container['output']))

      valProtocol[valIteration][10] = valProtocol[valIteration][10] + torch.mean(valCodes)
      valProtocol[valIteration][11] = valProtocol[valIteration][11] + torch.std(valCodes)
      valProtocol[valIteration][12] = valProtocol[valIteration][12] + torch.std(logvar.output)

      valProtocol[valIteration][13] = valProtocol[valIteration][13] + criterion.criteria[1].output
      valProtocol[valIteration][15] = valProtocol[valIteration][15] + criterion.criteria[2].output

      valProtocol[valIteration][17] = valProtocol[valIteration][17] + torch.mean(torch.abs(valPreds:narrow(2, 1, 1) - container['output']:narrow(2, 1, 1)))
      valProtocol[valIteration][19] = valProtocol[valIteration][19] + torch.mean(torch.abs(valPreds:narrow(2, 2, 1) - container['output']:narrow(2, 2, 1)))

      local valOccPreds = valPreds:narrow(2, 1, 1)
      valOccPreds[valOccPreds:gt(0.5)] = 1
      valOccPreds[valOccPreds:lt(0.5)] = 0
      valProtocol[valIteration][21] = valProtocol[valIteration][21] + torch.mean(torch.abs(valOccPreds - container['output']:narrow(2, 1, 1)))

      local valSDFPreds = valPreds:narrow(2, 2, 1)
      valSDFPreds[valSDFPreds:gt(0)] = 0
      valSDFPreds[valSDFPreds:lt(0)] = 1
      valProtocol[valIteration][23] = valProtocol[valIteration][23] + torch.mean(torch.abs(valSDFPreds - container['output']:narrow(2, 1, 1)))

      local valPreds = decoder:forward(valCodes)
      local f = criterion:forward(valPreds, {container['points'], container['space'], container['points'], container['space']})
      valProtocol[valIteration][3] = valProtocol[valIteration][3] + f
      valProtocol[valIteration][5] = valProtocol[valIteration][5] + KLD.loss

      valProtocol[valIteration][7] = valProtocol[valIteration][7] + torch.mean(torch.abs(valPreds - container['output']))

      accValPreds = appendTensor(accValPreds, valPreds)
    end

    valProtocol[valIteration][1] = t
    for i = 2, valProtocol:size(2) do
      valProtocol[valIteration][i] = valProtocol[valIteration][i] / valNumBatches
    end

    accValPreds = accValPreds:narrow(1, 1, valN)
    accValCodes = accValCodes:narrow(1, 1, valN)

    local predFile = config['test_directory'] .. t .. '_predictions.h5'
    lib.utils.writeHDF5(predFile, accValPreds)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': saved ' .. predFile)

    local codeFile = config['test_directory'] .. t .. '_codes.h5'
    lib.utils.writeHDF5(codeFile, accValCodes)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': saved ' .. codeFile)

    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': codes ' .. valProtocol[valIteration][10] .. ' / ' .. valProtocol[valIteration][11])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': logvar ' .. valProtocol[valIteration][12])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation loss ' .. valProtocol[valIteration][2])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': KLD loss ' .. valProtocol[valIteration][4])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation error ' .. valProtocol[valIteration][6] .. ' / ' .. valProtocol[valIteration][7])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation (abs+occ) ' .. valProtocol[valIteration][17])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation (abs+sdf) ' .. valProtocol[valIteration][19])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation (abs+occ+thresh) ' .. valProtocol[valIteration][21])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec ..': validation (abs+sdf+thresh) ' .. valProtocol[valIteration][23])

    local nll = valProtocol[valIteration][2] + valProtocol[valIteration][3]
    if nll < earlyStopError then
      earlyStopError = nll
      local earlyStopIteration = torch.Tensor(1)

      local earlyStopFile = config['early_stop_file']
      earlyStopIteration[1] = t
      lib.utils.writeHDF5(earlyStopFile, earlyStopIteration)
      print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. earlyStopFile)

      local modelFile = config['test_directory'] .. t .. '.dat'
      torch.save(modelFile, model)
      print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': snapshot ' .. modelFile)
    end
  end
end

-- Save the accumulated loss.
lib.utils.writeHDF5(config['train_protocol_file'], protocol)
print('[Training] protocol ' .. config['train_protocol_file'])

lib.utils.writeHDF5(config['val_protocol_file'], valProtocol)
print('[Training] protocol ' .. config['val_protocol_file'])

modelFile = config['vae_aml_parameters']['model_file']
torch.save(modelFile, model)
print('[Training] snapshot ' .. modelFile)