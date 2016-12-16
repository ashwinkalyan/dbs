local paths = require 'paths'
local utils = require 'misc.utils'
local vis_utils = {}
debug = require 'debug'

-- Vocab keys are strings, not integers, for some reason.
-- Javascript doesn't like that (it uses 0 based indexing).
local function convert_vocab(vocab)
    local new_vocab = {}
    for k,v in pairs(vocab) do
        new_vocab[tonumber(k)] = v
    end
    return new_vocab
end

function vis_utils.export_vis(final_beams, vis_table, img_path, vocab, T, vis_dir, end_token, primetext) 
    local json = require('rapidjson')
    beams = {}
    final_logprobs = {}
    final_divscores = {}
    num_beams_per_group = {}
    for i = 1,#final_beams do
        beam_group = final_beams[i]
        num_beams_per_group[i] = #beam_group
        for j = 1,#beam_group do
            beam = beam_group[j]
            idx = (i-1) * #beam_group + j
            seq = torch.reshape(beam.seq, beam.seq:size(1), 1)
            beams[idx] = primetext..' '..table.concat(net_utils.decode_sequence(vocab, seq))
            final_logprobs[idx] = beam['unaug_logp']
            final_divscores[idx] = beam['logp']
        end
    end
    vis_table['final_beams'] = beams
    vis_table['final_logprobs'] = final_logprobs
    vis_table['final_divscores'] = final_divscores
    -- The ivocab terminology comes from char-rnn, where the meaning of the vocab
    -- map is reversed from what it is here.
    vis_table['ivocab'] = convert_vocab(vocab)
    vis_table['T'] = T
    vis_table['num_beams_per_group'] = num_beams_per_group
    vis_table['img_path'] = img_path
    vis_table['img_fname'] = paths.basename(img_path)
    vis_table['end_token'] = end_token
    vis_table['prime_text'] = primetext

    json_fname = paths.concat(vis_dir, 'data.json')
    json.dump(vis_table, json_fname)

    new_img_path = paths.concat(vis_dir, vis_table['img_fname'])
    copy_cmd = 'cp "' .. img_path .. '" "' .. new_img_path .. '"'
    os.execute(copy_cmd)
end

return vis_utils
