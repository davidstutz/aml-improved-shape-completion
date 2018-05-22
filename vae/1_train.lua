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

-- Load configuration.
configFile = 'config.json'
if arg[1] and lib.utils.fileExists(arg[1]) then
  configFile = arg[1]
end

print('[Training] reading ' .. configFile)
config = lib.utils.readJSON(configFile)

config['prior_parameters']['model_file'] = config['base_directory'] .. config['prior_parameters']['model_file']
config['base_directory'] = config['base_directory'] .. '/prior/'
config['test_directory'] = config['base_directory'] .. config['test_directory']
config['train_protocol_file'] = config['base_directory'] .. config['train_protocol_file']
config['val_protocol_file'] = config['base_directory'] .. config['val_protocol_file']
config['early_stop_file'] = config['base_directory'] .. config['early_stop_file']

keys = {
  'prior_training_outputs',
  'prior_training_sdf_outputs',
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
print('[Training] reading ' .. config['prior_training_outputs'])
outputs_1 = lib.utils.readHDF5(config['prior_training_outputs'])
print('[Training] reading ' .. config['prior_training_sdf_outputs'])
outputs_2 = lib.utils.readHDF5(config['prior_training_sdf_outputs'])
outputs = torch.cat({outputs_1:float(), outputs_2:float()}, 2)

-- !
maxLTSDF = torch.max(outputs_2)
print('[Training] max LTSDF ' .. maxLTSDF)

-- Load data for validation.
print('[Training] reading ' .. config['validation_outputs'])
valOutputs_1 = lib.utils.readHDF5(config['validation_outputs'])
print('[Training] reading ' .. config['validation_sdf_outputs'])
valOutputs_2 = lib.utils.readHDF5(config['validation_sdf_outputs'])
valOutputs = torch.cat({valOutputs_1:float(), valOutputs_2:float()}, 2)

-- Create snapshot directory.
print('[Training] creating ' .. config['test_directory'])
if not lib.utils.directoryExists(config['test_directory']) then
  lib.utils.makeDirectory(config['test_directory'])
end

-- Load random codes.
print('[Training] reading ' .. config['codes_file'])
testCodes = lib.utils.readHDF5(config['codes_file'])
testCodes = testCodes:cuda()
testCodes = testCodes:transpose(1, 2)

-- Some assertions to check configuration.
assert(#config['channels'] == #config['kernel_sizes'])

-- For later simplicity.
N = outputs:size()[1]
height = outputs:size()[3]
width = outputs:size()[4]
depth = outputs:size()[5]

-- Set up config for model.
autoEncoderConfig = lib.variationalAutoEncoder.config
autoEncoderConfig.height = height
autoEncoderConfig.width = width
autoEncoderConfig.depth = depth
autoEncoderConfig.code = config['code']
autoEncoderConfig.channels = config['channels']
autoEncoderConfig.kernelSizes = config['kernel_sizes']
autoEncoderConfig.pooling = config['pooling']
autoEncoderConfig.poolingSizes = config['pooling_sizes']

if config['transfer'] == 'relu' then
  autoEncoderConfig.transfer = nn.ReLU
else
  assert(false, 'invalid transfer')
end

autoEncoderConfig.encoderStrides = config['encoder_strides']
autoEncoderConfig.decoderStrides = config['decoder_strides']
autoEncoderConfig.decoderKernelSizes = config['decoder_kernel_sizes']
autoEncoderConfig.transfers = config['transfers']
autoEncoderConfig.normalizations = config['normalizations']
autoEncoderConfig.dropouts = config['dropouts']
if config['dropout_probability'] then
  autoEncoderConfig.dropoutProbability = config['dropout_probability']
end
autoEncoderConfig.dataChannels = outputs:size(2)
autoEncoderConfig.outputChannels = outputs:size(2)
--autoEncoderConfig.printDimensions = true
--autoEncoderConfig.checkNaN = true

-- Set up the auto-encoder.
model = nn.Sequential()
model, context = lib.variationalAutoEncoder.autoEncoder(model, autoEncoderConfig)
context['decoder']:remove(#context['decoder'].modules)

nonLinearities = nn.PerChannelNonLinearity()
nonLinearities.layers = {nn.Sigmoid(), nn.Identity()}
context['decoder']:add(nonLinearities)

encoder = context['encoder']
decoder = context['decoder']

KLD = context['KLD']
KLD.sizeAverage = config['prior_parameters']['size_average']
KLD.lambda = config['prior_parameters']['prior_weight']

if KLD.sizeAverage then
  print('[Training] size average and lambda ' .. KLD.lambda)
else
  print('[Training] NO size average and lambda ' .. KLD.lambda)
end

-- !
if config['prior_parameters']['centering'] then
  local mean = torch.mean(outputs, 1)
  centering = nn.Center(mean)
  uncentering = nn.UnCenter(mean)
  encoder:insert(centering, 1)
  decoder:insert(uncentering, #decoder.modules)
  print('[Training] using centering')
end

if config['prior_parameters']['noise_level'] ~= nil and config['prior_parameters']['noise_level'] > 0 then
  noise = nn.PerChannelNonLinearity()
  noise.layers = {nn.SaltPepperNoise(), nn.GaussianNoise()}
  noise.layers[1].p = config['prior_parameters']['noise_level']
  noise.layers[2].p = config['prior_parameters']['noise_level']/2
  encoder:insert(noise, 1)
  print('[Training] using noise ' .. config['prior_parameters']['noise_level'])
end

mean = context['mean']
logvar = context['logVar']

-- Initialize weights.
lib.init(model, config['prior_parameters']['weight_initialization'], config['prior_parameters']['weight_value'],
  config['prior_parameters']['bias_initialization'], config['prior_parameters']['bias_value'])
model = model:cuda()
print(model)

-- Criterion.
occCriterion = nn.BCECriterion()
occCriterion.sizeAverage = config['prior_parameters']['size_average']
occCriterion = occCriterion:cuda()

sdfCriterion = nn.FixedVarianceGaussianNLLCriterion()
sdfCriterion.sizeAverage = config['prior_parameters']['size_average']
sdfCriterion.logvar = 0
sdfCriterion = sdfCriterion:cuda()

criterion = nn.PerChannelCriterion()
criterion.criteria = {occCriterion, sdfCriterion}
criterion.weights = {1, 1 }
criterion = criterion:cuda()

-- Learning hyperparameters.
optimizer = optim.sgd
if config['prior_parameters']['optimizer'] then
  if config['prior_parameters']['optimizer'] == 'sgd' then
    print('[Training] using sgd')
  elseif config['prior_parameters']['optimizer'] == 'adam' then
    optimizer = optim.adam
    print('[Training] using adam')
  elseif config['prior_parameters']['optimizer'] == 'rmsprop' then
    optimizer = optim.rmsprop
    print('[Training] using rmsprop')
  else
    assert(false)
  end
end

batchSize = config['prior_parameters']['batch_size']
learningRate = config['prior_parameters']['learning_rate']
momentum = config['prior_parameters']['momentum']
weightDecay = config['prior_parameters']['weight_decay']
epochs = config['prior_parameters']['epochs']
iterations = epochs*math.floor(N/batchSize)
minimumLearningRate = config['prior_parameters']['minimum_learning_rate']
learningRateDecay = config['prior_parameters']['decay_learning_rate']
maximumMomentum = config['prior_parameters']['maximum_momentum']
momentumDecay = config['prior_parameters']['decay_momentum']
decayIterations = config['prior_parameters']['decay_iterations']
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
protocol = torch.Tensor(iterations, 17):fill(0)
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

  protocol[t][17] = 0
  if dataAugmentation then
    for b = 1, batchSize do
      local r = math.random()
      if r < dataAugmentation then
        local translate_h = torch.random(1, 4) - 2
        local translate_w = torch.random(1, 6) - 3
        local translate_d = torch.random(1, 4) - 2
        lib.translate_mirror(_output:narrow(1, b, 1), output:narrow(1, b, 1), translate_h, translate_w, translate_d)
        protocol[t][17] = protocol[t][17] + 1
      else
        output:narrow(1, b, 1):copy(_output:narrow(1, b, 1))
      end
    end
  else
    output = _output
  end

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
    local pred = model:forward(output)
    local f = criterion:forward(pred, output)

    protocol[t][2] = f
    protocol[t][10] = criterion.criteria[1].output
    protocol[t][11] = criterion.criteria[2].output
    protocol[t][3] = KLD.loss

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

    protocol[t][14] = torch.mean(mean.output)
    protocol[t][15] = torch.std(mean.output)
    protocol[t][16] = torch.mean(logvar.output)

    -- Add the Kullback-Leibler loss.
    f = f + KLD.loss

    gradParameters:clamp(-1, 1)
    return f, gradParameters
  end

  -- Save learning rate and momentum in protocol.
  protocol[t][7] = learningRate
  protocol[t][8] = momentum
  protocol[t][9] = KLD.lambda

  -- Update state with learning rate and momentum.
  adamState = adamState or {
    learningRate = learningRate,
    momentum = momentum,
    --beta1 = momentum,
    learningRateDecay = 0 -- will be done manually below
  }

  optimizer(feval, parameters, adamState)
  local time = os.date("*t")

  -- Report a smoothed loss instead of batch loss.
  if t%lossIterations == 0 then

    -- Compute losses over the alst config['loss_iterations'] iterations.
    local smoothedLoss = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 2, 1))
    local smoothedKLD = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 3, 1))
    local smoothedError = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 5, 1))
    local smoothedDataAugmentation = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 17, 1))

    local smoothedMean = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 14, 1))
    local smoothedStd = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 15, 1))
    local smoothedLogVar = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 16, 1))

    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ':'
            .. ' (' .. lib.utils.format_num(smoothedDataAugmentation) .. ')'
            .. ' [' .. lib.utils.format_num(smoothedLoss) .. ' | ' .. lib.utils.format_num(smoothedKLD) .. ']'
	        .. ' [' .. lib.utils.format_num(smoothedError) .. ']'
            .. ' [' .. lib.utils.format_num(smoothedMean) .. ' | ' .. lib.utils.format_num(smoothedStd).. ' | ' .. lib.utils.format_num(smoothedLogVar) .. ']')
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
    local accValMeanPreds = nil
    local accValCodes = nil

    -- iteration, loss x2, KLD loss x2, error x2, thresh error x2, mean mean, var mean, mean logvar
    for b = 0, valNumBatches do
      local batchStart = b*valBatchSize + 1
      local batchLength = math.min((b + 1)*valBatchSize - b*valBatchSize, valN - b*valBatchSize)

      if batchLength == 0 then
        break;
      end

      local output = valOutputs:narrow(1, batchStart, batchLength)
      output = output:cuda()

      local valPreds = model:forward(output)
      local valCodes = mean.output
      local valLogVar = logvar.output

      local f = criterion:forward(valPreds, output)
      valProtocol[valIteration][2] = valProtocol[valIteration][2] + f
      valProtocol[valIteration][4] = valProtocol[valIteration][4] + KLD.loss

      valProtocol[valIteration][15] = valProtocol[valIteration][15] + criterion.criteria[1].output
      valProtocol[valIteration][16] = valProtocol[valIteration][16] + criterion.criteria[2].output

      valProtocol[valIteration][6] = valProtocol[valIteration][6] + torch.mean(torch.abs(valPreds - output))
      valProtocol[valIteration][13] = valProtocol[valIteration][13] + torch.mean(torch.abs(valPreds:narrow(2, 1, 1) - output:narrow(2, 1, 1)))
      valProtocol[valIteration][14] = valProtocol[valIteration][14] + torch.mean(torch.abs(valPreds:narrow(2, 2, 1) - output:narrow(2, 2, 1)))

      valProtocol[valIteration][10] = valProtocol[valIteration][10] + torch.mean(valCodes)
      valProtocol[valIteration][11] = valProtocol[valIteration][11] + torch.std(valCodes)
      valProtocol[valIteration][12] = valProtocol[valIteration][12] + torch.mean(valLogVar)

      accValPreds = appendTensor(accValPreds, valPreds)
      accValCodes = appendTensor(accValCodes, valCodes)

      local valPreds = decoder:forward(valCodes)

      local f = criterion:forward(valPreds, output)
      valProtocol[valIteration][3] = valProtocol[valIteration][3] + f
      valProtocol[valIteration][5] = valProtocol[valIteration][5] + KLD.loss

      valProtocol[valIteration][7] = valProtocol[valIteration][7] + torch.mean(torch.abs(valPreds - output))

      accValMeanPreds = appendTensor(accValMeanPreds, valPreds)
    end

    accValPreds = accValPreds:narrow(1, 1, valN)
    accValCodes = accValCodes:narrow(1, 1, valN)
    accValMeanPreds = accValMeanPreds:narrow(1, 1, valN)

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

    local meanPredFile = config['test_directory'] .. t .. '_mean_predictions.h5'
    lib.utils.writeHDF5(meanPredFile, accValMeanPreds)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. meanPredFile)

    local codeFile = config['test_directory'] .. t .. '_codes.h5'
    lib.utils.writeHDF5(codeFile, accValCodes)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. codeFile)

    local randomFile = config['test_directory'] .. t .. '_random.h5'
    local randomPreds = decoder:forward(testCodes)
    randomPreds = randomPreds:float()
    lib.utils.writeHDF5(randomFile, randomPreds)
    print('[Training] ' .. t .. '|' .. iterations .. '|' .. time.hour .. ':' .. time.min .. ':' .. time.sec .. ': saved ' .. randomFile)

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
modelFile = config['prior_parameters']['model_file']
torch.save(modelFile, model)
print('[Training] snapshot ' .. modelFile)
