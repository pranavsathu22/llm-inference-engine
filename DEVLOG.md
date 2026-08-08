# Devlog

## Day 1 — tokenizer + data loading  (2026-07-11, ~1h)
-imported the data from tiny shakespeare
-had an issue where I was return a list of characters instead of a joined string

## Day 1b — get_batch + venv setup  (2026-07-12, ~45m)
- Loaded the tokens from the text and mapped the targets (y) by shifting one
- Ran into errors with pytorch dimensions, learned the specifics pf torch.stack 
  along each dimension
- stack needs N equal-shaped units to line up along a new axis

## Day 2 — attention deep dive (no code yet)  (2026-07-12, ~1h)
- Walked through the self-attention mechanism by hand to see how the dimensions 
  and associated tensors mapped out in each step of the process
    -specifically why only the inner two dimensions had to be transposed to multply query by key vectors
- Finally figured what the (T, T) tensor actuall meant
- Explored cross attention and its differences

# Day 3
  - had an issue with torch.cat, wasn't using the right dimensions
  - was calling each head.forward(), but that isn't consistent with how nn.module works
  - learned why there needs to be projection layer after concatening the output of the different heads

# Day 4 - 7/21/2026
  - took some time to understand feedforward networks, will actually code tmrw
  - learned the intution behind having multiple passes for attention
  - figured out why you need Layernorm / multiple bloks

# Day 5
  - wrote the forward passing logic as well as final LM_head to get raw scores for each token in vocab
  - understood the underlying mechanism(dot product to see similarily) to determine which token is at each posiiton in the sequence
  - Had an issue with weights initialization, so added in a custom initializer

# Day 6
  - setup the training loop for the llm, pretty straightforward other than figuring out what parameters to pass in
  - accidentally called loss.backward() before the forward pass that actually produces loss, crashed every run until I reordered it

# Day 7
  - went into depth of how the decoding process works and gets appended to the generation
  - fixed a couple of bugs
    - softmax was missing a dimension argument 
    - made multiple dimension errors, namely one being that I passed in the entire out tensor instead of taking one batch
    - first output with naive generation: 
        
        That, I a rut! fathrair, me! ance
        sostat exstry reeny the mut moranter hold alive
        The cle; whaven of with it love souleence father, like,
        I are How'd byforksings ingle with fienow somio.'

        DUKE VINCENV

# Day 8
  - added KV caching through the entire stack
  - calculated offset and appended new kv onto cached kv in head.forward
  - generate_cached prefills once, then decodes one token at a time
  - Main bug: Changing GPT.forward's return from (logits, loss) to (logits, loss, new_caches) broke 10 call sites across train.py and multiple test files 

# Day 10
  - implemented int8 affine quantization for the KV cache (quantize_int8/dequantize_int8),
    per-token grouping (one scale/zero_point per token, reducing over head_size)
  - had to clamp scale to a minimum (1e-8) to avoid divide-by-zero when a group's
    values are all identical -- verified the resulting huge zero_point still
    reconstructs exactly instead of producing NaN
  - measured (not projected) real memory savings via tensor_bytes: only 2.0x on an
    8-wide group, not the naive "int8 = 4x smaller" claim, because scale/zero_point
    are stored as full float32 and that overhead is proportionally large on small groups
  - swept group size 8 -> 2048 and confirmed the ratio follows
    (group_size*4)/(group_size+8), approaching 4x asymptotically but never reaching it
  - the real number for this project's actual model (head_size=32, from n_embd=128/
    n_head=4) is ~3.2x -- the honest figure to cite later, not "4x"
