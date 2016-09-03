local utils = require 'misc.utils'
local net_utils = {}

-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size))
  cnn_part:add(backend.ReLU(true))
  return cnn_part
end

-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

function net_utils.beam_step(logprobsf,beam_size,t,divm,vocab_size,seq_length,beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag)
--INPUTS:
--logprobsf: probabilities augmented after diversity
--beam_size: obvious
--t        : time instant
--beam_seq : tensor contanining the beams
--beam_seq_logprobs: tensor contanining the beam logprobs
--beam_logprobs_sum: tensor contanining joint logprobs
--OUPUTS:
--beam_seq
--beam_seq_logprobs
--beam_logprobs_sum
  print('beam_size:' .. beam_size)
  local beam_seq = beam_seq_table[divm]
  local beam_seq_logprobs = beam_seq_logprobs_table[divm]
  local beam_logprobs_sum = beam_logprobs_sum_table[divm]
  t = t- divm -1  
  local function compare(a,b) return a.p > b.p end -- used downstream
  ys,ix = torch.sort(logprobsf,2,true)
  candidates = {}
  cols = math.min(beam_size,ys:size()[2])
  rows = beam_size
  if t == 1 then rows = 1 end
  for c=1,cols do -- for each column (word, essentially)
    for q=1,rows do -- for each beam expansion
    --compute logprob of expanding beam q with word in (sorted) position c
      local local_logprob = ys[{ q,c }]
      local candidate_logprob = beam_logprobs_sum[q] + local_logprob
      table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
    end
  end
  table.sort(candidates, compare)
  new_state = net_utils.clone_list(state)
--local beam_seq_prev, beam_seq_logprobs_prev
  if t > 1 then
  --we''ll need these as reference when we fork beams around
    beam_seq_prev = beam_seq[{ {1,t-1}, {} }]:clone()
    beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-1}, {} }]:clone()
  end
  for vix=1,beam_size do
    v = candidates[vix]
  --fork beam index q into index vix
    if t > 1 then
      beam_seq[{ {1,t-1}, vix }] = beam_seq_prev[{ {}, v.q }]
      beam_seq_logprobs[{ {1,t-1}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
    end
  --rearrange recurrent states
    for state_ix = 1,#new_state do
--  copy over state in previous beam q to new beam at vix
      new_state[state_ix][vix] = state[state_ix][v.q]
    end
--append new end terminal at the end of this beam
    beam_seq[{ t, vix }] = v.c -- c'th word is the continuation
    beam_seq_logprobs[{ t, vix }] = v.r -- the raw logprob here
    beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam
    if v.c == vocab_size+1 or t == seq_length then
      if done_beams_flag[divm][vix] == 0 then 
      print('finished with beam '..'time: '..t)
      --END token special case here, or we reached the end.
      -- add the beam to a set of done beams
        done_beams_table[divm][vix] = {seq = beam_seq_table[divm][{ {}, vix }]:clone(), logps = beam_seq_logprobs_table[divm][{ {}, vix }]:clone(),p = beam_logprobs_sum_table[divm][vix]}
        done_beams_flag[divm][vix] = 1
      end
    end
  end
  if new_state then state = new_state end
  print('done with 1 beam_step\n')
  return beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag
end

return net_utils
