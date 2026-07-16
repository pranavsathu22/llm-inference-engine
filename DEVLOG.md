# Devlog

3–5 honest lines a day. What I built, what broke, how I figured it out. This is the
comprehension check (if I can't write it, I didn't understand it) and the interview
prep. Keep the confusion in — the debugging *is* the record of real work.

## Day 0 — scaffold  (2026-07-10)
- Built: repo skeleton, empty stubs, this log.
- Broke / confused me: —
- Figured out: —
- Still fuzzy on: —

## Day 1 — tokenizer + data loading  (2026-07-11, ~1h)
-imported the data from tiny shakespeare
-had an issue where I was return a list of characters instead of a joined string

## Day 1b — get_batch + venv setup  (2026-07-12, ~45m)
- Loaded the tokens from the text and mapped the targets (y) by shifting one
- Ran into errors with pytorch dimensions, learned the specifics pf torch.stack 
  along each dimension
- stack needs N *equal-shaped* units to line up along a new axis

## Day 2 — attention deep dive (no code yet)  (2026-07-12, ~1h)
- Walked through the self-attention mechanism by hand to see how the dimensions 
  and associated tensors mapped out in each step of the process
    -specifically why only the inner two dimensions had to be transposed to multply query by key vectors
- Finally figured what the (T, T) tensor actuall meant
- Explored cross attention and its differences

