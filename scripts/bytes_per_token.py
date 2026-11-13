"""Bytes read from VRAM per decoded token = every weight tensor except token_embd
(only one row of it is read), unless the model has no output.weight (tied head)."""
import sys, json, glob, os
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
from gguf import GGUFReader
out = {}
for f in sorted(glob.glob("models/*.gguf")):
    r = GGUFReader(f)
    names = {t.name for t in r.tensors}
    tied = "output.weight" not in names
    total = 0
    for t in r.tensors:
        b = int(t.n_bytes)
        if t.name == "token_embd.weight" and not tied:
            b = b // int(t.shape[1])          # one row per token
        total += b
    out[os.path.basename(f)[:-5]] = {"bytes_per_token": total, "file_bytes": os.path.getsize(f), "tied": tied}
json.dump(out, open("results/bytes_per_token.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:32s} per-token {v['bytes_per_token']/1e6:7.1f} MB  file {v['file_bytes']/1e6:7.1f} MB  tied={v['tied']}")
