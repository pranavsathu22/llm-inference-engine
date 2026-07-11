"""Day 1 — fetch the corpus (TinyShakespeare, ~1 MB) into data/input.txt.

It's a ~5-line urllib download. Left as a stub so you type it — you should never have a
line in this repo you couldn't have written yourself.

URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Save to: data/input.txt  (gitignored — regenerable)
"""

from __future__ import annotations

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = "data/input.txt"


def main() -> None:
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(URL, DEST)
    print(f"downloaded {DEST} ({os.path.getsize(DEST)} bytes)")


if __name__ == "__main__":
    main()
