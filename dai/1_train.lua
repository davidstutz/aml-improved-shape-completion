-- Train an auto-encoder using config.json.

require('torch')
require('nn')
require('optim')
require('hdf5')
require('cunn')
require('lfs')

--- Append the tensor tensor to the tensor acc which may initially be nil.
local function appendTensor(acc, tensor)
  if acc == nil then
    acc = tensor:float()
  else
    acc = torch.cat(acc, tensor:float(), 1)
  end

  return acc
end

package.path = package.path .. ';../?/th/init.lua'
lib = require('lib')

cnncomplete = {}
include('0_model.lua')

-- Load configuration.
configFile = 'config.json'
if arg[1] and lib.utils.fileExists(arg[1]) then
  configFile = arg[1]
end

print('[Training] reading ' .. configFile)
config = lib.utils.readJSON(configFile)

config['sup_parameters']['model_file'] = config['base_directory'] .. config['sup_parameters']['model_file']
config['base_directory'] = config['base_directory'] .. '/sup/'
config['test_directory'] = config['base_directory'] .. config['test_directory']
config['train_protocol_file'] = config['base_directory'] .. config['train_protocol_file']
config['val_protocol_file'] = config['base_directory'] .. config['val_protocol_file']
config['early_stop_file'] = config['base_directory'] .. config['early_stop_file']

keys = {
  'inference_training_inputs',
  'inference_training_sdf_inputs',
  'inference_training_outputs',
  'inference_training_sdf_outputs',
  'validation_inputs',
  'validation_sdf_inputs',
  'validation_outputs',
  'validation_sdf_outputs',
}

for i, key in pairs(keys) do
  config[key] = config['data_directory'] .. '/' .. config[key]
  if not lib.utils.fileOrDirectoryExists(config[key]) then
    print('[Error] file or directory ' .. config[key] .. ' does not exist')
    os.exit()
  end
end

-- Load data for training.
print('[Training] reading ' .. config['inference_training_inputs'])
inputs_1 = lib.utils.readHDF5(config['inference_training_inputs'])
print('[Training] reading ' .. config['inference_training_sdf_inputs'])
inputs_2 = lib.utils.readHDF5(config['inference_training_sdf_inputs'])

nObservations = inputs_1:size(2)
inputs_1 = inputs_1:resize(inputs_1:size(1)*inputs_1:size(2), 1, inputs_1:size(3), inputs_1:size(4), inputs_1:size(5))
inputs_2 = inputs_2:resize(inputs_2:size(1)*inputs_2:size(2), 1, inputs_2:size(3), inputs_2:size(4), inputs_2:size(5))
inputs = torch.cat({inputs_1:float(), inputs_2:float()}, 2)

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

assert(inputs:size(1) == outputs:size(1))
assert(inputs:size(2) == outputs:size(2))
assert(inputs:size(2) == 2)
assert(inputs:size(3) == outputs:size(3))
assert(inputs:size(4) == outputs:size(4))
assert(inputs:size(5) == outputs:size(5))

-- Load data for validation.
print('[Training] reading ' .. config['validation_inputs'])
valInputs_1 = lib.utils.readHDF5(config['validation_inputs'])
print('[Training] reading ' .. config['validation_sdf_inputs'])
valInputs_2 = lib.utils.readHDF5(config['validation_sdf_inputs'])

valInputs_1 = valInputs_1:resize(valInputs_1:size(1)*valInputs_1:size(2), 1, valInputs_1:size(3), valInputs_1:size(4), valInputs_1:size(5))
valInputs_2 = valInputs_2:resize(valInputs_2:size(1)*valInputs_2:size(2), 1, valInputs_2:size(3), valInputs_2:size(4), valInputs_2:size(5))
valInputs = torch.cat({valInputs_1:float(), valInputs_2:float()}, 2)

print('[Training] reading ' .. config['validation_outputs'])
valOutputs_1 = lib.utils.readHDF5(config['validation_outputs'])
print('[Training] reading ' .. config['validation_sdf_outputs'])
valOutputs_2 = lib.utils.readHDF5(config['validation_sdf_outputs'])

valOutputs_1 = valOutputs_1:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*valOutputs_1:size(1), 1, valOutputs_1:size(3), valOutputs_1:size(4), valOutputs_1:size(5))
valOutputs_2 = valOutputs_2:repeatTensor(1, nObservations, 1, 1, 1):resize(nObservations*valOutputs_2:size(1), 1, valOutputs_2:size(3), valOutputs_2:size(4), valOutputs_2:size(5))
valOutputs = torch.cat({valOutputs_1:float(), valOutputs_2:float()}, 2)

assert(valInputs:size(1) == valOutputs:size(1))
assert(valInputs:size(2) == valOutputs:size(2))
assert(valInputs:size(2) == 2)
assert(valInputs:size(3) == valOutputs:size(3))
assert(valInputs:size(4) == valOutputs:size(4))
assert(valInputs:size(5) == valOutputs:size(5))

-- Create snapshot directory.
print('[Training] creating ' .. config['test_directory'])
if not lib.utils.directoryExists(config['test_directory']) then
  lib.utils.makeDirectory(config['test_directory'])
end

-- Some assertions to check configuration.
assert(#config['channels'] == #config['kernel_sizes'])

-- For later simplicity.
N = outputs:size()[1]
height = outputs:size()[3]
width = outputs:size()[4]
depth = outputs:size()[5]

if width == 54 then
  model = cnncomplete.Models.low54(cnncomplete.opts)
elseif width == 72 then
  model = cnncomplete.Models.med72(cnncomplete.opts)
elseif width == 108 then
  model = cnncomplete.Models.high108(cnncomplete.opts)
elseif width == 32 then
  model = cnncomplete.Models.low32(cnncomplete.opts)
elseif width == 48 then
  model = cnncomplete.Models.med48(cnncomplete.opts)
elseif width == 64 then
  model = cnncomplete.Models.med64(cnncomplete.opts)
else
  print('[Error] invalid resolution or not suitable model in 0_model.lua found')
  os.exit()
end

model = model:cuda()
print(model)

-- Criterion.
occCriterion = nn.BCECriterion()
occCriterion.sizeAverage = config['sup_parameters']['size_average']
occCriterion = occCriterion:cuda()

sdfCriterion = nn.FixedVarianceGaussianNLLCriterion()
sdfCriterion.sizeAverage = config['sup_parameters']['size_average']
sdfCriterion.logvar = 0
sdfCriterion = sdfCriterion:cuda()

criterion = nn.PerChannelCriterion()
criterion.criteria = {occCriterion, sdfCriterion }
criterion.weights = {1, 1 }
criterion = criterion:cuda()

-- Learning hyperparameters.
optimizer = optim.adam
batchSize = config['sup_parameters']['batch_size']
learningRate = config['sup_parameters']['learning_rate']
momentum = config['sup_parameters']['momentum']
weightDecay = config['sup_parameters']['weight_decay']
epochs = config['sup_parameters']['epochs']
iterations = epochs*math.floor(N/batchSize)
minimumLearningRate = config['sup_parameters']['minimum_learning_rate']
learningRateDecay = config['sup_parameters']['decay_learning_rate']
maximumMomentum = config['sup_parameters']['maximum_momentum']
momentumDecay = config['sup_parameters']['decay_momentum']
decayIterations = config['sup_parameters']['decay_iterations']
lossIterations = config['loss_iterations']
snapshotIterations = config['snapshot_iterations']
testIterations = config['test_iterations']
if not snapshotIterations then
  snapshotIterations = 2*testIterations
end
earlyStopError = 1e20

dataAugmentation = 0.5
if dataAugmentation > 0 then
  print('[Training] data augmentation')
end

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

-- Saves the loss.
protocol = torch.Tensor(iterations, 14):fill(0)
-- Will save: iteration, loss, KLD loss, error, thresh error, learning rate, momentum, lambda
valProtocol = torch.Tensor(math.floor(iterations/testIterations) + 1, 16):fill(0)
-- Wil save: iteration, loss x2, KLD loss x2, error x2, thresh error x2, mean mean, var mean, mean logvar

-- Main training loop.
for t = 1, iterations do

  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()

  local _output = outputs:index(1, shuffle)
  local output = torch.Tensor(batchSize, 2, height, width, depth)

  local _input = inputs:index(1, shuffle)
  local input = torch.Tensor(batchSize, 2, height, width, depth)

  protocol[t][14] = 0
  if dataAugmentation then
    for b = 1, batchSize do
      local r = math.random()
      if r < dataAugmentation then
        local translate_h = torch.random(1, 4) - 2
        local translate_w = torch.random(1, 6) - 3
        local translate_d = torch.random(1, 4) - 2
        lib.translate_mirror(_output:narrow(1, b, 1), output:narrow(1, b, 1), translate_h, translate_w, translate_d)
        lib.translate_mirror(_input:narrow(1, b, 1), input:narrow(1, b, 1), translate_h, translate_w, translate_d)
        protocol[t][14] = protocol[t][14] + 1
      else
        output:narrow(1, b, 1):copy(_output:narrow(1, b, 1))
        input:narrow(1, b, 1):copy(_input:narrow(1, b, 1))
      end
    end
  else
    output = _output
  end

  input = input:cuda()
  output = output:cuda()
  protocol[t][1] = t

  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
  local feval = function(x)

    -- Get new parameters.
    if x ~= parameters then
      parameters:copy(x)
    end

    -- Reset gradients
    gradParameters:zero()

    -- Evaluate function on mini-batch.
    local pred = model:forward(input)
    local f = criterion:forward(pred, output)

    protocol[t][2] = f
    protocol[t][10] = criterion.criteria[1].output
    protocol[t][11] = criterion.criteria[2].output

    -- Estimate df/dW.
    local df_do = criterion:backward(pred, output)
    model:backward(output, df_do)

    -- Weight decay:
    if weightDecay > 0 then
      weightDecayLoss = weightDecay * torch.norm(parameters,2)^2/2
      protocol[t][4] = weightDecayLoss

      f = f + weightDecayLoss
      gradParameters:add(parameters:clone():mul(weightDecay))
    end

    protocol[t][5] = torch.mean(torch.abs(pred - output))
    protocol[t][12] = torch.mean(torch.abs(pred:narrow(2, 1, 1) - output:narrow(2, 1, 1)))
    protocol[t][13] = torch.mean(torch.abs(pred:narrow(2, 2, 1) - output:narrow(2, 2, 1)))

    gradParameters:clamp(-1, 1)
    -- return f and df/dX
    return f, gradParameters
  end

  -- Save learning rate and momentum in protocol.
  protocol[t][7] = learningRate
  protocol[t][8] = momentum

  -- Update state with learning rate and momentum.
  adamState = adamState or {
    learningRate = learningRate,
    momentum = momentum,
    learningRateDecay = 0 -- will be done manually below
  }

  optimizer(feval, parameters, adamState)
  local time = os.date("*t")

  -- Report a smoothed loss instead of batch loss.
  if t%lossIterations == 0 then

    -- Compute losses over the alst config['loss_iterations'] iterations.
    local smoothedLoss = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 2, 1))
    local smoothedWeightDecay = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 4, 1))
    local smoothedError = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 5, 1))
    local smoothedDataAugmentation = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 14, 1))

    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ':'
            .. ' (' .. lib.utils.format_num(smoothedDataAugmentation) .. ')'
            .. ' [' .. lib.utils.format_num(smoothedLoss) .. ']'
            .. ' [' .. lib.utils.format_num(smoothedError) .. ']')
  end

  -- Decay learning rate and KLD weight, do this before resetting all smoothed
  -- statistics.
  if t%decayIterations == 0 then
    learningRate = math.max(minimumLearningRate, learningRate*learningRateDecay)
    momentum = math.min(maximumMomentum, momentum*momentumDecay)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': learning rate ' .. learningRate)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': momentum ' .. momentum)
  end

  -- Validate on validation set.
  if t%testIterations == 0 or t == 1 or t == iterations then

    -- In case the validation set gets to large.
    local valN = valOutputs:size(1)
    local valBatchSize = batchSize
    local valNumBatches = math.floor(valN/valBatchSize)
    local valIteration = math.floor(t/testIterations) + 1

    local accValPreds = nil

    -- iteration, loss x2, KLD loss x2, error x2, thresh error x2, mean mean, var mean, mean logvar
    for b = 0, valNumBatches do
      local batchStart = b*valBatchSize + 1
      local batchLength = math.min((b + 1)*valBatchSize - b*valBatchSize, valN - b*valBatchSize)

      if batchLength == 0 then
        break;
      end

      local input = valInputs:narrow(1, batchStart, batchLength)
      local output = valOutputs:narrow(1, batchStart, batchLength)

      input = input:cuda()
      output = output:cuda()

      local valPreds = model:forward(input)

      local f = criterion:forward(valPreds, output)
      valProtocol[valIteration][2] = valProtocol[valIteration][2] + f

      valProtocol[valIteration][15] = valProtocol[valIteration][15] + criterion.criteria[1].output
      valProtocol[valIteration][16] = valProtocol[valIteration][16] + criterion.criteria[2].output

      valProtocol[valIteration][6] = valProtocol[valIteration][6] + torch.mean(torch.abs(valPreds - output))
      valProtocol[valIteration][13] = valProtocol[valIteration][13] + torch.mean(torch.abs(valPreds:narrow(2, 1, 1) - output:narrow(2, 1, 1)))
      valProtocol[valIteration][14] = valProtocol[valIteration][14] + torch.mean(torch.abs(valPreds:narrow(2, 2, 1) - output:narrow(2, 2, 1)))

      accValPreds = appendTensor(accValPreds, valPreds)
    end

    accValPreds = accValPreds:narrow(1, 1, valN)

    valProtocol[valIteration][1] = t
    for i = 2, valProtocol:size(2) do
      valProtocol[valIteration][i] = valProtocol[valIteration][i] / valNumBatches
    end

    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': codes (mean) ' .. valProtocol[valIteration][10] .. ' / ' .. valProtocol[valIteration][11])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': codes (logvar) ' .. valProtocol[valIteration][12])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': KLD loss ' .. valProtocol[valIteration][4])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': validation loss ' .. valProtocol[valIteration][2])
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': validation error (abs) ' .. valProtocol[valIteration][6] .. ' / ' .. valProtocol[valIteration][7])

    local predFile = config['test_directory'] .. t .. '_predictions.h5'
    lib.utils.writeHDF5(predFile, accValPreds)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. predFile)

    local nll = valProtocol[valIteration][3] + valProtocol[valIteration][5]
    if nll < earlyStopError then
      earlyStopError = nll
      local earlyStopIteration = torch.Tensor(1)

      local earlyStopFile = config['early_stop_file']
      earlyStopIteration[1] = t
      lib.utils.writeHDF5(earlyStopFile, earlyStopIteration)
      print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. earlyStopFile)

      local modelFile = config['test_directory'] .. 'early_stop.dat'
      torch.save(modelFile, model)
      print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': snapshot ' .. modelFile)
    end
  end

  if t%snapshotIterations == 0 and t > 1 then
    local modelFile = config['test_directory'] .. t .. '.dat'
    torch.save(modelFile, model)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': snapshot ' .. modelFile)
  end
end

-- Save the accumulated loss.
protocolFile = config['train_protocol_file']
lib.utils.writeHDF5(protocolFile, protocol)
print('[Training] protocol ' .. protocolFile)

protocolFile = config['val_protocol_file']
lib.utils.writeHDF5(protocolFile, valProtocol)
print('[Training] protocol ' .. protocolFile)

-- Save the model.
modelFile = config['sup_parameters']['model_file']
torch.save(modelFile, model)
print('[Training] snapshot ' .. modelFile)