require('torch')
require('nn')

--- @class GaussianNoise
local GaussianNoise, GaussianNoiseParent = torch.class('nn.GaussianNoise', 'nn.Module')

--- Initialize.
-- @param p probability of salt and pepper
function GaussianNoise:__init(p)
  self.p = p or 0.05
end

--- Compute forward pass, i.e. threshold to 1 at 0.1.
-- @param input layer input
-- @param output
function GaussianNoise:updateOutput(input)
  if self.train ~= false and self.p > 0 then
    local rand = torch.randn(input:size())*self.p

    if input.__typename == 'torch.CudaTensor' then
      rand = rand:cuda()
    end

    self.output = input + rand
  else
    self.output = input
  end

  return self.output
end

--- Compute the backward pass.
-- @param input original input
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input
function GaussianNoise:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end