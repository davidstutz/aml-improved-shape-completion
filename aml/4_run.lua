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

-- https://github.com/torch/cutorch
if cutorch.isCachingAllocatorEnabled() then
  print('[Training] caching allocator enabled')
end

configFile = arg[1]
config = lib.utils.readJSON(configFile)
config['vae_aml_parameters']['model_file'] = config['base_directory'] .. '/' .. config['vae_aml_parameters']['model_file']
config['base_directory'] = config['base_directory'] .. '/vae_aml/'
config['test_directory'] = config['base_directory'] .. config['test_directory'] .. '/'

keys = {
  'validation_inputs',
  'validation_sdf_inputs',
}

for i, key in pairs(keys) do
  config[key] = config['data_directory'] .. '/' .. config[key]
  if not lib.utils.fileOrDirectoryExists(config[key]) then
    print('[Error] file or directory ' .. config[key] .. ' does not exist')
    os.exit()
  end
end

print('[Training] creating ' .. config['test_directory'])
if not lib.utils.directoryExists(config['test_directory']) then
  lib.utils.makeDirectory(config['test_directory'])
end

print('[Training] loading ' .. config['vae_aml_parameters']['model_file'])
model = torch.load(config['vae_aml_parameters']['model_file'])
decoder = model.modules[2]
print(model)

print('[Training] reading ' .. config['validation_inputs'])
valInputs_1 = lib.utils.readHDF5(config['validation_inputs'])
print('[Training] reading ' .. config['validation_sdf_inputs'])
valInputs_2 = lib.utils.readHDF5(config['validation_sdf_inputs'])

-- ! First reshape inputs.
valInputs_1 = valInputs_1:resize(valInputs_1:size(1)*valInputs_1:size(2), 1, valInputs_1:size(3), valInputs_1:size(4), valInputs_1:size(5))
valInputs_2 = valInputs_2:resize(valInputs_2:size(1)*valInputs_2:size(2), 1, valInputs_2:size(3), valInputs_2:size(4), valInputs_2:size(5))

valInputs = torch.cat({valInputs_1:float(), valInputs_2:float()}, 2)

local valN = valInputs:size(1)
print('[Training] processing ' .. valN .. ' observations')
local valBatchSize = 16
local valNumBatches = math.floor(valN/valBatchSize)

local accValPreds = nil
local accValMeanPreds = nil

-- iteration, loss x2, KLD loss x2, error x2, thresh error x2, mean mean, var mean, mean logvar
for b = 0, valNumBatches do
  local batchStart = b*valBatchSize + 1
  local batchLength = math.min((b + 1)*valBatchSize - b*valBatchSize, valN - b*valBatchSize)

  if batchLength == 0 then
    break;
  end

  local input = valInputs:narrow(1, batchStart, batchLength):cuda()

  local valPreds = model:forward(input)
  accValPreds = appendTensor(accValPreds, valPreds)

  local valCodes = model.modules[1].modules[#model.modules[1].modules - 2].modules[1].output
  local valPreds = decoder:forward(valCodes)
  accValMeanPreds = appendTensor(accValMeanPreds, valPreds)
  print('[Training] processed ' .. b .. '/' .. valNumBatches)
end

predFile = config['test_directory'] .. '0_predictions.h5'
lib.utils.writeHDF5(predFile, accValMeanPreds)
print('[Training] wrote ' .. predFile)