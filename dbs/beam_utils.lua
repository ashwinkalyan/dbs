require 'dbs.util.misc' -- for table.slice
require 'torchx'
local beam_utils = {}

-- function to compare
function beam_utils.compare(a,b) return a.p > b.p end

function beam_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

function beam_utils.vocab_map(ivocab, div_ivocab, ix)

    return ivocab[div_ivocab[ix]]
end
-- function computes the weighting factor for the diversity
function add_diversity(beam_seq_table,logprobsf,t,divm,opt,bdash)
  
    if opt.div_msr ==1 and t >0 and t<opt.cutoff_msr then
 
         local div_checkpoint = opt.msr_div_checkpoint
         local div_vocab = div_checkpoint.vocab
         local div_ivocab ={}
         local ivocab = {}
         div_protos = div_checkpoint.protos
         div_protos.rnn:evaluate()
         for c,i in pairs(div_vocab) do div_ivocab[i] = c end
         local CharSplitLMMinibatchLoader = require 'msr_div_utils.CharSplitLMMinibatchLoader'
         for c,i in pairs(opt.vocab) do ivocab[i] = c end
  
         for i=1,beam_seq_table[1]:size(2) do -- this is equivalent to "for each beam"
             local text_until_now = net_utils.decode_sequence(opt.vocab, beam_seq_table[1][{{},{i}}])[1]
  
             seed_text = text_until_now  

             -- initialize the rnn state for the diversity model
             local div_current_state, div_state_predict_index
             local div_model = div_checkpoint.opt.model

             local current_state
             current_state = {}
             for L = 1,div_checkpoint.opt.num_layers do
                 -- c and h for all layers
                 local h_init = torch.zeros(1, div_checkpoint.opt.rnn_size):cuda()
                 if opt.gpuid >= 0  then h_init = h_init:cuda() end

                 table.insert(current_state, h_init:clone())
                 if div_checkpoint.opt.model == 'lstm' then
                     table.insert(current_state, h_init:clone())
                 end
             end
             state_size = #current_state
             if string.len(seed_text) > 0 then 
                 --print('seeding with ' .. seed_text)
                 --print('--------------------------')
                 local tokens = {}
                 for c in CharSplitLMMinibatchLoader.tokens(seed_text, opt.word_level == 1) do
                     if div_vocab[c] == nil then c = c:lower() end
                     tokens[#tokens + 1] = c
                 end
                 --tokens[#tokens + 1] = '.'
                 sent_prob = 0


                 for _, c in ipairs(tokens) do --todo: word_level should be stored in checkpoint

                     local idx = div_vocab[c]
                     if idx ~= nil then
                         prev_char = torch.Tensor{idx}
                         if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end

                         local lst = div_protos.rnn:forward{prev_char, unpack(current_state)}
                         -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
                         current_state = {}
                         for i=1,state_size do table.insert(current_state, lst[i]) end
                         prediction = lst[#lst] -- last element holds the log probabilities
                      end
                 end
             else
                 prediction = torch.Tensor(1, #div_ivocab):fill(1)/(#div_ivocab)
                    
             end

             for j=1,prediction[1]:size(1) do

                 if beam_utils.vocab_map(ivocab,div_ivocab, j) ~= nil then
                     
                     logprobsf[i][beam_utils.vocab_map(ivocab,div_ivocab, j)] = logprobsf[i][beam_utils.vocab_map(ivocab,div_ivocab, j)] - opt.lambda_msr * prediction[1][j] 
                end
            
             end
       end
       
      
       return logprobsf     
    
else
    local local_time = t - divm + 1
	local unaug_logprobsf = logprobsf:clone()
  for prev_choice = 1, divm-1 do
    prev_decisions = beam_seq_table[prev_choice][local_time]
    for sub_beam = 1,bdash do
      for prev_labels = 1,bdash do 
        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - opt.lambda*div_utils.cumulative_div(beam_seq_table,t,divm,opt.divmode)
      end
    end
  end
	return unaug_logprobsf
end
end
-- does one step of beam_step
local function beam_step(logprobsf,unaug_logprobsf,beam_size,t,beam_seq,beam_seq_logprobs,beam_logprobs_sum,state)
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
  local ys,ix = torch.sort(logprobsf,2,true)
  local candidates = {}
  local cols = math.min(beam_size,ys:size()[2])
  local rows = beam_size
  if t == 1 then rows = 1 end
  for c=1,cols do -- for each column (word, essentially)
    for q=1,rows do -- for each beam expansion
    --compute logprob of expanding beam q with word in (sorted) position c
      local_logprob = ys[{ q,c }]
      candidate_logprob = beam_logprobs_sum[q] + local_logprob
			local_unaug_logprob = unaug_logprobsf[{q,ix[{q,c}]}]
      table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_unaug_logprob })
    end
  end
  table.sort(candidates, beam_utils.compare)
  new_state = beam_utils.clone_list(state)
--local beam_seq_prev, beam_seq_logprobs_prev
  if t > 1 then
  --we''ll need these as reference when we fork beams around
    beam_seq_prev = beam_seq[{ {1,t-1}, {} }]:clone()
    beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-1}, {} }]:clone()
  end
  for vix=1,beam_size do
    v = candidates[vix]
    v.kept = true
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
  end
  state = new_state
  return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates
end

-- implements beam search
-- calls beam_step and returns the final set of beams
function beam_utils.beam_search(init_params, opt)
  local bdash = opt.B / opt.M
  local init_state = init_params[1]
  local init_logprobs = init_params[2]
  local vis_iterations = opt.vis_iterations
  local export_vis = (vis_iterations ~= nil)
  local state_table = {}
  local beam_seq_table = {}
  local beam_seq_logprobs_table = {}
  local beam_logprobs_sum_table = {}
  -- INITIALIZATIONS
  for i=1,opt.M do
    state_table[i] = {}
  end
  for i=1,opt.M do
    beam_seq_table[i] = torch.Tensor(opt.T, bdash):zero()
    if opt.use_gpu then
        beam_seq_table[i] = beam_seq_table[i]:cuda()
    end
  end
  for i=1,opt.M do
    beam_seq_logprobs_table[i] = torch.Tensor(opt.T, bdash):zero()
    if opt.use_gpu then
        beam_seq_logprobs_table[i] = beam_seq_logprobs_table[i]:cuda()
    end
  end
  for i=1,opt.M do
    beam_logprobs_sum_table[i] = torch.zeros(bdash)
    if opt.use_gpu then
        beam_logprobs_sum_table[i] = beam_logprobs_sum_table[i]:cuda()
    end
  end
  -- logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
  done_beams_table = {}
  for i=1,opt.M do
    done_beams_table[i] = {}
  end
  state = {}
  for h =1,opt.state_size do
    state[h] = torch.zeros(bdash, opt.rnn_size)
    if opt.use_gpu then
        state[h] = state[h]:cuda()
    end
  end
  for h=1,opt.state_size do
    for b = 1, bdash do
      state[h][b] = init_state[h]:clone()
    end
  end
  for i=1,opt.M do
    state_table[i] = beam_utils.clone_list(state)
  end
  logprobs_table = {}
  for i=1,opt.M do 
    logprobs_table[i] = torch.zeros(bdash, init_logprobs:size()[2])
    if opt.use_gpu then
        logprobs_table[i] = logprobs_table[i]:cuda()
    end
    for j = 1,bdash do logprobs_table[i][j] = init_logprobs:clone() end
  end
  -- END INIT

  for t=1,opt.T+opt.M-1 do
    local vis_candidates = {}
    for divm = 1,opt.M do 
      if t>= divm and t<= opt.T+divm-1 then 
        -- add diversity
        logprobsf = logprobs_table[divm]
        -- suppress UNK tokens.
        logprobsf[{{},logprobsf:size()[2]-1}] =  logprobsf[{ {}, logprobsf:size()[2]-1}] - 1000
        local unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,opt,bdash)
        -- infer new beams
        beam_seq_table[divm],
        beam_seq_logprobs_table[divm],
        beam_logprobs_sum_table[divm],
        state_table[divm],
        candidates_divm = beam_step(logprobsf,
																		unaug_logprobsf,
                                    bdash,
                                    t-divm+1,
                                    beam_seq_table[divm],
                                    beam_seq_logprobs_table[divm],
                                    beam_logprobs_sum_table[divm],
                                    state_table[divm])

        -- cache candidates for visualization
        local candidates_kept_for_divm = {}
        if export_vis then
          for ci = 1,#candidates_divm do
            c = candidates_divm[ci]
            c.cid = #vis_candidates
            c.t = t
            c.t_prev = t - 1
            c.cid_prev = (divm - 1 - math.max(0, t - opt.T - 1)) * bdash * bdash + c.q
            c.local_t = t - divm + 1
            c.divm = divm
            table.insert(vis_candidates, c)
            if c.kept then
              table.insert(candidates_kept_for_divm, c)
            end
          end
          assert(#candidates_kept_for_divm == bdash)
        end

        -- if time's up... or if end token is reached then copy beams
        for vix=1,bdash do
        local logp_add = 0
          local is_first_end_token = ((beam_seq_table[divm][{ {},vix}][t-divm+1] == opt.end_token) and (torch.eq(beam_seq_table[divm][{ {},vix}],opt.end_token):sum()==1))
          local final_time_without_end_token = ((t == opt.T+divm-1) and (torch.eq(beam_seq_table[divm][{ {},vix}],opt.end_token):sum()==0))
          if opt.div_msr ==1 then 
              length = torch.find(beam_seq_table[divm][{{},vix}]:clone(),0)[1] or 16 
              logp_add = opt.phi_msr*length
          end
          if is_first_end_token or final_time_without_end_token then
            final_beam = {
              seq = beam_seq_table[divm][{ {}, vix }]:clone(),
              length = torch.find(beam_seq_table[divm][{{},vix}]:clone(),0)[1] or 16,
              logps = beam_seq_logprobs_table[divm][{ {}, vix }]:clone(),
							unaug_logp = beam_seq_logprobs_table[divm][{ {}, vix}]:sum(),
              logp = beam_logprobs_sum_table[divm][vix] +logp_add
            }
            if export_vis then
              final_beam['candidate'] = candidates_kept_for_divm[vix]
            end
            table.insert(done_beams_table[divm], final_beam)
          end
          -- don't continue beams from finished sequences
          if is_first_end_token then
            beam_logprobs_sum_table[divm][vix] = -1000
          end
        end

        -- move this group one step forward in time
        it = beam_seq_table[divm][t-divm+1]
        out = opt.gen_logprobs(it,state_table[divm])
        logprobs_table[divm] = out[#out]:clone()
        temp_state = {}
        for i=1,opt.state_size do table.insert(temp_state, out[i]) end
        state_table[divm] = beam_utils.clone_list(temp_state)
      end
    end
    if export_vis then
      table.insert(vis_iterations, vis_candidates)
    end
  end
  local function compare_beam(a,b) return a.logp > b.logp end
  for i =1,opt.M do
    table.sort(done_beams_table[i],compare_beam)
    done_beams_table[i] = table.slice(done_beams_table[i], 1, bdash, 1)
    if export_vis then
      for j = 1,#(done_beams_table[i]) do
        local beam = done_beams_table[i][j]
        beam.candidate.kept_forever = true
      end
    end
  end
  return done_beams_table
end

return beam_utils
