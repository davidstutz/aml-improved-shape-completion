-- Data augmentation.

--- Translate the volume in the given dimension.
-- Set new values according to mirrored old values at the borders.
-- @param input tensor to translate
-- @param output output tensor to write to
-- @param dim dimension
-- @param t_dim translation along dim
lib.translate_dim_mirror = function(input, output, dim, t_dim)
  local size_dim = input:size(dim)

  if t_dim >= 0 then
    output:narrow(dim, t_dim + 1, size_dim - t_dim):copy(input:narrow(dim, 1, size_dim - t_dim))

    -- Fill "new values" created through translating by mirroring old values.
    for t = 1, t_dim do
      output:narrow(dim, t, 1):copy(input:narrow(dim, t_dim, 1))
    end
  end
  if t_dim < 0 then
    output:narrow(dim, 1, size_dim + t_dim):copy(input:narrow(dim, -t_dim + 1, size_dim + t_dim))
    for t = t_dim, -1 do
      output:narrow(dim, size_dim + t + 1, 1):copy(input:narrow(dim, size_dim, 1))
    end
  end
end

--- Translate the volume in the given dimension.
-- Set new values to a fixed value.
-- @param input tensor to translate
-- @param output output tensor to write to
-- @param dim dimension
-- @param t_dim translation along dim
-- @param default default value
lib.translate_dim_fixed = function(input, output, dim, t_dim, default)
  local size_dim = input:size(dim)

  if t_dim >= 0 then
    output:narrow(dim, t_dim + 1, size_dim - t_dim):copy(input:narrow(dim, 1, size_dim - t_dim))

    -- Fill "new values" with default value.
    for t = 1, t_dim do
      output:narrow(dim, t, 1):fill(default)
    end
  end
  if t_dim < 0 then
    output:narrow(dim, 1, size_dim + t_dim):copy(input:narrow(dim, -t_dim + 1, size_dim + t_dim))
    for t = t_dim, -1 do
      output:narrow(dim, size_dim + t + 1, 1):fill(default)
    end
  end
end

--- Translate a N x B x H x W x D tensor in height/width/depth dimensions.
-- @param input input tensor to translate
-- @param output pre-allocate doutput tensor to write to
-- @param t_height translation in height
-- @param t_width translation in width
-- @param t_depth translation in depth
lib.translate_mirror = function(input, output, t_height, t_width, t_depth)
  assert(math.abs(t_height) < input:size(3))
  assert(math.abs(t_width) < input:size(4))
  assert(math.abs(t_depth) < input:size(5))

  local temp = input:clone()
  lib.translate_dim_mirror(temp, output, 3, t_height)

  temp:copy(output)
  output:fill(0)
  lib.translate_dim_mirror(temp, output, 4, t_width)

  temp:copy(output)
  output:fill(0)
  lib.translate_dim_mirror(temp, output, 5, t_depth)
end

--- Translate a N x B x H x W x D tensor in height/width/depth dimensions.
-- @param input input tensor to translate
-- @param output pre-allocate doutput tensor to write to
-- @param t_height translation in height
-- @param t_width translation in width
-- @param t_depth translation in depth
-- @param default default value
lib.translate_fixed = function(input, output, t_height, t_width, t_depth, default)
  assert(math.abs(t_height) < input:size(3))
  assert(math.abs(t_width) < input:size(4))
  assert(math.abs(t_depth) < input:size(5))

  local temp = input:clone()
  lib.translate_dim_fixed(temp, output, 3, t_height, default)

  temp:copy(output)
  output:fill(0)
  lib.translate_dim_fixed(temp, output, 4, t_width, default)

  temp:copy(output)
  output:fill(0)
  lib.translate_dim_fixed(temp, output, 5, t_depth, default)
end