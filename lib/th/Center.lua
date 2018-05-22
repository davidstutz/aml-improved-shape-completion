require('torch')
require('nn')

--- @class Center
local Center, CenterParent = torch.class('nn.Center', 'nn.Module')

--- Initialize.
function Center:__init(mean)
  self.mean = mean
end

--- Print dimensions of last layer.
-- @param input output of last layer
-- @return unchanged output of last layer
function Center:updateOutput(input)
  assert(self.mean)

  if input:type() == 'torch.CudaTensor' then
    self.mean = self.mean:cuda()
  end

  self.output = input - torch.repeatTensor(self.mean, input:size(1), 1, 1, 1, 1)

  return self.output
end

--- Print the gradients of the next layer.
-- @param input original input of last layer
-- @param gradOutput gradients of next layer
-- @return unchanged gradients of next layer
function Center:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput

  return self.gradInput
end