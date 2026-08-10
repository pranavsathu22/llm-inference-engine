# Devlog

## 2026-07-11 -- tokenizer + data loading  (~1h)
-imported the data from tiny shakespeare
-had an issue where I was return a list of characters instead of a joined string

## 2026-07-13 -- get_batch + venv setup  (~45m)
- Loaded the tokens from the text and mapped the targets (y) by shifting one
- Ran into errors with pytorch dimensions, learned the specifics pf torch.stack 
  along each dimension
- stack needs N equal-shaped units to line up along a new axis

## 2026-07-14 -- attention deep dive (no code yet)  (~1h)
- Walked through the self-attention mechanism by hand to see how the dimensions 
  and associated tensors mapped out in each step of the process
    -specifically why only the inner two dimensions had to be transposed to multply query by key vectors
- Finally figured what the (T, T) tensor actuall meant
- Explored cross attention and its differences

## 2026-07-20
  - had an issue with torch.cat, wasn't using the right dimensions
  - was calling each head.forward(), but that isn't consistent with how nn.module works
  - learned why there needs to be projection layer after concatening the output of the different heads

## 2026-07-21
  - took some time to understand feedforward networks, will actually code tmrw
  - learned the intution behind having multiple passes for attention
  - figured out why you need Layernorm / multiple bloks

## 2026-07-27
  - wrote the forward passing logic as well as final LM_head to get raw scores for each token in vocab
  - understood the underlying mechanism(dot product to see similarily) to determine which token is at each posiiton in the sequence
  - Had an issue with weights initialization, so added in a custom initializer

## 2026-07-28
  - setup the training loop for the llm, pretty straightforward other than figuring out what parameters to pass in
  - accidentally called loss.backward() before the forward pass that actually produces loss, crashed every run until I reordered it

## 2026-07-29
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

## 2026-08-02
  - added KV caching through the entire stack
  - calculated offset and appended new kv onto cached kv in head.forward
  - generate_cached prefills once, then decodes one token at a time
  - Main bug: Changing GPT.forward's return from (logits, loss) to (logits, loss, new_caches) broke 10 call sites across train.py and multiple test files 

## 2026-08-04
  - implemented top_k and top_p (nucleus) filtering in sample_next
  - top_k: threshold at the k-th largest logit per row, mask everything below it to
    -inf so softmax assigns those tokens exactly 0 probability
  - top_p: sort descending, cumulative-sum the probabilities, keep the smallest
    prefix whose mass reaches p -- shift the removal mask right by one so the token
    that crosses the threshold is kept, always keep the top-1 token even if it alone
    exceeds p
  - verified top_k=1 is exactly greedy (always picks the argmax); top_p with a small
    p correctly excludes the clearly-lowest-probability candidate
  - found generate_cached's prefill step still hardcoded temperature=1.0/top_k=None/
    top_p=None instead of forwarding **sample_kwargs -- same bug class as the one
    found on 2026-08-02, just missed on the very first sampled token

## 2026-08-07
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
  - wrote bench.py: tokens/sec (naive vs cached) across generation lengths, KV-cache
    bytes (fp32/fp16/int8) across sequence length, reused the 2026-07-29 loss curve
  - time_generation does a warmup pass + median of repeats, not a single cold run
  - important correction: the earlier informal 2026-08-02 check (single, no warmup)
    showed cached LOSING (0.52x) -- that was measurement noise, not real. The proper
    benchmark (warmup + 3-repeat median, swept across lengths 8-63) shows cached
    consistently winning, 1.13x-1.94x. Lesson: single uncontrolled timing runs can
    straight up lie to you -- exactly the kind of thing that matters for real
    benchmarking, not just this toy project
  - fp16/int8 ratio came out to a constant 1.60x at every sequence length (makes
    sense: head_size=32 never changes, so the per-group overhead fraction doesn't
    either) -- consistent with the 3.2x fp32/int8 finding above, since fp16 is just
    half of fp32's bytes: 3.2/2 = 1.6, exactly matching what got measured
  - polish pass: removed stale TODO comments and dead code across every module, fixed
    type hints that had gone stale since the caching change, rewrote the README
    results section with the real numbers/plots instead of the day-1 placeholder,
    dropped the build-plan checklist now that it's done, retrained to regenerate the
    loss curve with an actual title on it (the original had none)
