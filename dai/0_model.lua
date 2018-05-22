require('torch')
require('nn')
require('nnx')
require('nngraph')

require('cutorch')
require('cunn')
require('cunnx')

package.path = package.path .. ';../?/th/init.lua'
lib = require('lib')
assert(cnncomplete)

local function mergeDefaults(tbl, defaultTbl)
    if defaultTbl then
        local newTbl = {}
        for k,v in pairs(tbl) do newTbl[k] = v end
        for k,v in pairs(defaultTbl) do
            if newTbl[k] == nil then newTbl[k] = v end
        end
        return newTbl
    else
        return tbl
    end
end

local function checkOpts(opts)
    opts = opts or {}
    opts = mergeDefaults(opts, {
        activation = 'ReLU',
        leakyReluSlope = 0.2,
        doBatchNorm = true,
        batchNormEps = 1e-3
    })
    return opts
end

local function addModulesToSeq(nnseq, modules)
    for _,mod in ipairs(modules) do
        nnseq:add(mod)
    end
    return nil
end

local function addModulesToGraph(node, modules)
    for _,mod in ipairs(modules) do
        node = mod(node)
    end
    return node
end

local SequentialMT = getmetatable(nn.Sequential)
local NodeMT = getmetatable(nngraph.Node)
local function addModules(x, modules)
    local mt = getmetatable(x)
    if mt == SequentialMT then
        return addModulesToSeq(x, modules)
    elseif mt == NodeMT then
        return addModulesToGraph(x, modules)
    else
        print(mt)
        error('addModules only accepts nn.Sequential or nngraph.Node inputs')
    end
end

local function activationModules(modules, opts)
    if opts.activation == 'ReLU' then
        table.insert(modules, nn.ReLU(true))
        print('nn.ReLU')
    elseif opts.activation == 'LeakyReLU' then
        table.insert(modules, nn.LeakyReLU(opts.leakyReluSlope, true))
        print('nn.LeakyReLU')
    elseif opts.activation == 'none' then
        print('no activation')
    else
        error('Unrecognized activation ' .. opts.activation)
    end
end

local function VolConvBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut, size, strides, pad)
        return function(x)
            local modules = {nn.VolumetricConvolution(nIn, nOut, size, size, size, strides[1], strides[3], strides[2], pad, pad, pad)}
            --table.insert(modules, nn.PrintDimensions())
            print('nn.VolumetricConvolution')
            if opts.doBatchNorm then
                table.insert(modules, nn.VolumetricBatchNormalization(nOut, opts.batchNormEps))
                print('nn.VolumetricBatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

local function VolUpConvBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut, sizes, strides, pad, extra)
        return function(x)
            --local modules = {nn.VolumetricFullConvolution(nIn, nOut, size, size, size, stride, stride, stride, pad, pad, pad, extra, extra, extra)}
            local modules = {nn.VolumetricFullConvolution(nIn, nOut, sizes[1], sizes[3], sizes[2], strides[1], strides[3], strides[2], pad, pad, pad) }
            --table.insert(modules, nn.PrintDimensions())
            print('nn.VolumetricFullConvolution')
            if opts.doBatchNorm then
                table.insert(modules, nn.VolumetricBatchNormalization(nOut, opts.batchNormEps))
                print('nn.VolumetricBatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

local function FullyConnectedBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut)
        return function(x)
            local modules = {nn.Linear(nIn, nOut) }
            --table.insert(modules, nn.PrintDimensions())
            print('nn.Linear')
            if opts.doBatchNorm then
                table.insert(modules, nn.BatchNormalization(nOut, opts.batchNormEps))
                print('nn.BatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

local function ModelLow(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc1 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, opts.nf, 4, {2, 2, 2}, 1)(vol)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 2, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {2, 2, 2}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 4, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {4, 4, 4}, {1, 1, 1}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*opts.nf, 2, {4, 4, 4}, {2, 2, 2}, 1, 0)(d4)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelHigh64(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc0 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, math.floor(opts.nf/6), 4, {2, 2, 2}, 1)(vol)
    local enc1 = VolConvBlock({ activation='LeakyReLU'})(math.floor(opts.nf/6), opts.nf, 4, {2, 2, 2}, 1)(enc0)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 2, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {2, 2, 2}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 4, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {4, 4, 4}, {1, 1, 1}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local dec4 = VolUpConvBlock()(2*opts.nf, math.floor(opts.nf/6), {4, 4, 4}, {2, 2, 2}, 1, 0)(d4)
    local d5 = nn.JoinTable(2)({dec4, enc0})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*math.floor(opts.nf/6), 2, {4, 4, 4}, {2, 2, 2}, 1, 0)(d5)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelMed48(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc0 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, math.floor(opts.nf/6), 4, {2, 2, 2}, 1)(vol)
    local enc1 = VolConvBlock({ activation='LeakyReLU'})(math.floor(opts.nf/6), opts.nf, 4, {2, 2, 2}, 1)(enc0)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 2, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {2, 2, 2}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 3, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {3, 3, 3}, {1, 1, 1}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 4, 4}, {2, 2, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local dec4 = VolUpConvBlock()(2*opts.nf, math.floor(opts.nf/6), {4, 4, 4}, {2, 2, 2}, 1, 0)(d4)
    local d5 = nn.JoinTable(2)({dec4, enc0})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*math.floor(opts.nf/6), 2, {4, 4, 4}, {2, 2, 2}, 1, 0)(d5)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelLow32(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc1 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, opts.nf, 4, {2, 3, 2}, 1)(vol)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 3, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {3, 3, 3}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 2, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {2, 2, 2}, {2, 2, 2}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {3, 3, 3}, {3, 3, 3}, 0, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 5, 4}, {2, 3, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*opts.nf, 2, {4, 5, 4}, {2, 3, 2}, 1, 0)(d4)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelLow54(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc1 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, opts.nf, 4, {2, 3, 2}, 1)(vol)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 3, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {3, 3, 3}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 2, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {2, 2, 2}, {2, 2, 2}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {3, 3, 3}, {3, 3, 3}, 0, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 5, 4}, {2, 3, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*opts.nf, 2, {4, 5, 4}, {2, 3, 2}, 1, 0)(d4)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelMed72(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc0 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, math.floor(opts.nf/6), 4, {2, 2, 2}, 1)(vol)
    local enc1 = VolConvBlock({ activation='LeakyReLU'})(math.floor(opts.nf/6), opts.nf, 4, {2, 2, 2}, 1)(enc0)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 3, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {2, 3, 2}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 2, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {2, 2, 2}, {2, 2, 2}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {4, 5, 4}, {2, 3, 2}, 1, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 5, 4}, {2, 3, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local dec4 = VolUpConvBlock()(2*opts.nf, math.floor(opts.nf/6), {4, 4, 4}, {2, 2, 2}, 1, 0)(d4)
    local d5 = nn.JoinTable(2)({dec4, enc0})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*math.floor(opts.nf/6), 2, {4, 4, 4}, {2, 2, 2}, 1, 0)(d5)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

local function ModelHigh108(opts)
    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc0 = VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, math.floor(opts.nf/6), 4, {2, 2, 2}, 1)(vol)
    local enc1 = VolConvBlock({ activation='LeakyReLU' })(math.floor(opts.nf/6), opts.nf, 4, {2, 3, 2}, 1)(enc0)
    local enc2 = VolConvBlock({ activation='LeakyReLU' })(opts.nf, 2*opts.nf, 4, {2, 3, 2}, 1)(enc1)
    local enc3 = VolConvBlock({ activation='LeakyReLU' })(2*opts.nf, 4*opts.nf, 4, {3, 3, 3}, 1)(enc2)
    local enc4 = VolConvBlock({ activation='LeakyReLU' })(4*opts.nf, 8*opts.nf, 2, {1, 1, 1}, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*opts.nf))
    --bottleneck:add(nn.PrintDimensions())
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    FullyConnectedBlock({ doBatchNorm=false })(8*opts.nf, 8*opts.nf)(bottleneck)
    bottleneck:add(nn.View(8*opts.nf, 1, 1, 1))
    --bottleneck:add(nn.PrintDimensions())
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4})
    local dec1 = VolUpConvBlock()(2*8*opts.nf, 4*opts.nf, {2, 2, 2}, {2, 2, 2}, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = VolUpConvBlock()(2*4*opts.nf, 2*opts.nf, {3, 3, 3}, {3, 3, 3}, 0, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = VolUpConvBlock()(2*2*opts.nf, opts.nf, {4, 5, 4}, {2, 3, 2}, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local dec4 = VolUpConvBlock()(2*opts.nf, math.floor(opts.nf/6), {4, 5, 4}, {2, 3, 2}, 1, 0)(d4)
    local d5 = nn.JoinTable(2)({dec4, enc0})
    local decoded = VolUpConvBlock({ activation='none', doBatchNorm=false })(2*math.floor(opts.nf/6), 2, {4, 4, 4}, {2, 2, 2}, 1, 0)(d5)

    local nonlinearity = nn.PerChannelNonLinearity()
    nonlinearity.layers = {nn.Sigmoid(), nn.Identity()}
    decoded = nonlinearity(decoded)

    return nn.gModule(input, {decoded})
end

cnncomplete.Model = ModelLow
cnncomplete.Models = {
    low54 = ModelLow54,
    med72 = ModelMed72,
    high108 = ModelHigh108,
    low32 = ModelLow32,
    med48 = ModelMed48,
    high64 = ModelHigh64,
}
cnncomplete.opts = {
    nf = 80,
}