require('torch')
require('nn')

--- @class SaltPepperNoise
local SaltPepperNoise, SaltPepperNoiseParent = torch.class('nn.SaltPepperNoise', 'nn.Module')

--- Initialize.
-- @param p probability of salt and pepper
function SaltPepperNoise:__init(p)
  self.p = p or 0.05
end

--- Compute forward pass, i.e. threshold to 1 at 0.1.
-- @param input layer input
-- @param output
function SaltPepperNoise:updateOutput(input)
  if self.train ~= false and self.p > 0 then
    local rand = torch.rand(input:size())

    if input.__typename == 'torch.CudaTensor' then
      rand = rand:cuda()
    end

    rand[rand:gt(self.p)] = 0
    rand[rand:lt(self.p)] = 1

    self.output = input:clone()
    self.output[rand:eq(1)] = 1 - self.output[rand:eq(1)]
  else
    self.output = input
  end

  return self.output
end

--- Compute the backward pass.
-- @param input original input
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input
function SaltPepperNoise:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end