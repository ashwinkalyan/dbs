torch = require('torch')
debug = require('debug')
require('util.misc') -- for table.slice

vocab = torch.load('data/vocab_from_checkpoint.t7')
ivocab = {}
for c, i in pairs(vocab) do
    ivocab[i] = c
end
ivocab = table.slice(ivocab, 1, 79, 1)
old_vocab = vocab
vocab = {}
for i, c in pairs(ivocab) do
    vocab[c] = i
end

torch.save('data/vocab_from_checkpoint_fixed.t7', vocab)

