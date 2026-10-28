"""Ollama single-stream decode at the given prompt sizes; one JSON line per request.

Usage: python3 ollama_decode.py WORDS [WORDS ...]   (mac-probe-1.md §4, step B)
"""

import json
import sys
import urllib.request

MODEL = "qwen3-8b-q4km"
URL = "http://127.0.0.1:11434/api/generate"
REPS = 5
# Ten distinct words, cycled: a single repeated word makes greedy decoding emit one token over and
# over, and Ollama aborts after 100 identical tokens in a row (llm/llama_server.go, v0.40.11).
CYCLE = "one two three four five six seven eight nine ten".split()


def generate(words, rep):
    # The run label up front makes every prompt distinct, so the prompt cache cannot skip the prefill.
    prompt = f"Run {rep}." + "".join(" " + CYCLE[i % len(CYCLE)] for i in range(words))
    body = {
        "model": MODEL,
        "prompt": prompt,
        "raw": True,  # no chat template: the prompt is exactly the filler
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 128, "temperature": 0, "seed": 0},
    }
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request))
    return {k: reply[k] for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")}


for words in map(int, sys.argv[1:]):
    generate(words, "warm-up")  # discarded: loads the model and settles the clocks
    for rep in range(REPS):
        row = generate(words, rep)
        # The first token comes free from the prefill; eval_duration spans the other n - 1 steps.
        row["tok_s"] = (row["eval_count"] - 1) / row["eval_duration"] * 1e9  # durations are nanoseconds
        row["tok_s_reported"] = row["eval_count"] / row["eval_duration"] * 1e9
        print(json.dumps({"words": words, "rep": rep, **row}), flush=True)
