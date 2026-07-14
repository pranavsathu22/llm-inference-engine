# Devlog

3–5 honest lines a day. What I built, what broke, how I figured it out. This is the
comprehension check (if I can't write it, I didn't understand it) and the interview
prep. Keep the confusion in — the debugging *is* the record of real work.

Template:

```
## Day N — <topic>  (YYYY-MM-DD, ~Nm)
- Built:
- Broke / confused me:
- Figured out:
- Still fuzzy on:
```

---

## Day 0 — scaffold  (2026-07-10)
- Built: repo skeleton, empty stubs, this log.
- Broke / confused me: —
- Figured out: —
- Still fuzzy on: —

## Day 1 — tokenizer + data loading  (2026-07-11, ~1h)
- Built: download script for TinyShakespeare (data/input.txt, 1,115,394 bytes);
  CharTokenizer (stoi/itos/vocab_size, encode, decode) round-trip verified against
  the corpus; data.py's load_tokens + train_val_split (90/10 contiguous split).
- Broke / confused me: decode's round-trip assert kept failing even though the
  printed output *looked* right character-for-character. download_data.py also
  originally crashed — `import urllib` doesn't pull in `urllib.request`, and I'd
  left a stray `raise NotImplementedError` after the download call so it never
  even got to print a confirmation.
- Figured out: decode was returning a `list[str]` instead of a joined string —
  `['F','i','r','s','t']` prints similarly to `"First"` but `==` says they're
  never equal since the types differ. Fixed with `"".join(...)`. For the download
  script: import `urllib.request` explicitly (submodules aren't auto-imported),
  and drop the leftover raise once the real logic is in.
- Still fuzzy on: get_batch's random-window sampling (haven't written it yet —
  picking that up next session) and the right import path for tokenizer.py from
  data.py once I actually wire them together.

## Day 1b — get_batch + venv setup  (2026-07-12, ~45m)
- Built: get_batch (random-window batch sampler); a venv (torch 2.13.0+cpu, Python
  3.14) so the pipeline actually runs; ran tokenizer + data.py end to end against
  the real corpus and confirmed y really is x shifted by one char, e.g.
  x[1]="\nFor he " -> y[1]="For he i".
- Broke / confused me: first pass at get_batch called torch.stack(...) inside the
  loop and never assigned the result, so it silently did nothing. Fixing just the
  assignment still broke on batch_size > 2 -- stacking a (2, block_size) tensor
  with a (block_size,) tensor errors, since stack requires every input to be the
  identical shape. Separately assumed torch didn't support Python 3.14 yet and was
  about to set up an older interpreter for nothing.
- Figured out: stack needs N *equal-shaped* units to line up along a new axis --
  once you stack two windows together they stop being "a single window" and
  become "a batch of 2," so you can't stack that against a lone window anymore.
  Fix: collect all batch_size windows into a plain list first, stack once at the
  end. And: torch 2.13.0 ships cp314 wheels already -- pip install just worked,
  no downgrade needed. Should have tried the install before assuming it'd fail.
- Still fuzzy on: nothing blocking -- ready for Day 2 (single attention head).
