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



