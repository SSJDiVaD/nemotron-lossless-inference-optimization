#!/usr/bin/env python3
"""Generate a synthetic coding-agent trace (Claude Code / Codex traffic shape).

Each session = [global shared prefix: agent instructions + tool schemas + repo overview]
             + [per-session repo context: a few source files]
             + N turns of "edit this function, reply with a diff", each turn appending
               a canned assistant diff so prompts are identical across endpoints.

Source text comes from the Python stdlib, so it is real code, deterministic, and needs no download.

Output: one JSON object per line:
  {"id": "...", "messages": [system, ...], "turns": [{"user": str, "canned_reply": str}, ...]}
"""
import argparse, json, os, random, re, sysconfig

CHARS_PER_TOKEN = 3.6  # rough for code; pass --tokenizer for exact counts

AGENT_INSTRUCTIONS = """You are an autonomous coding agent operating inside a user's repository.
You can read files, search the codebase, run shell commands, and edit files using the tools provided.
Rules:
- Make the smallest change that fully solves the task. Do not refactor unrelated code.
- Preserve the existing code style, naming, and comments.
- When asked for an edit, reply ONLY with a unified diff against the current file contents.
- Never invent APIs; if unsure, read the relevant file first.
- Keep explanations to one sentence unless the user asks for more.
"""

TOOL_NAMES = ["read_file", "write_file", "edit_file", "list_dir", "glob", "grep", "run_shell",
              "run_tests", "git_diff", "git_status", "git_log", "web_fetch", "todo_write",
              "ask_user", "search_symbols"]


def tool_schemas():
    tools = []
    for name in TOOL_NAMES:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name.replace('_', ' ').capitalize()}. Use this tool when you need to "
                               f"{name.split('_')[0]} as part of completing the task. Returns structured "
                               f"output or an error string. Paths are relative to the repository root.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "Target path relative to the repo root."},
                    "pattern": {"type": "string", "description": "Optional glob or regex pattern."},
                    "max_results": {"type": "integer", "description": "Upper bound on returned items."},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds."}},
                    "required": ["path"]},
            }})
    return json.dumps(tools, indent=1)


def load_corpus(seed):
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("test", "tests", "idlelib", "site-packages", "__pycache__"))
        for f in sorted(filenames):
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                try:
                    txt = open(p, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                if 2_000 < len(txt) < 60_000 and re.search(r"^def \w+\(", txt, re.M):
                    files.append((os.path.relpath(p, root), txt))
    random.Random(seed).shuffle(files)
    return files


def make_counter(tokenizer_name):
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    return lambda s: int(len(s) / CHARS_PER_TOKEN)


def take_files(corpus, idx, budget, count):
    """Append files from corpus[idx:] until the token budget is reached."""
    parts, used, tries = [], 0, 0
    while used < budget * 0.9 and tries < len(corpus):
        # Wrap around: sessions may reuse files. That creates no spurious prefix-cache hits, because a hit needs
        # an identical PREFIX and each session's files come in a different order after the shared prefix.
        path, txt = corpus[idx % len(corpus)]
        idx += 1
        tries += 1
        block = f"<file path=\"{path}\">\n{txt}\n</file>\n"
        cost = count(block)
        if used + cost > budget:   # skip files that would overshoot; keeps prompts near the target length
            continue
        parts.append((path, txt, block))
        used += cost
    return parts, idx


TASKS = ["add input validation that raises ValueError on invalid arguments",
         "add type hints to the signature",
         "handle the empty-input edge case explicitly",
         "replace the loop with an equivalent comprehension where it is clearer",
         "add a docstring describing parameters and return value",
         "make the function robust to None arguments",
         "add a fast path for the common case"]


def canned_diff(path, txt, func, rng):
    lines = txt.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(f"def {func}(")), 0)
    ctx = lines[start:start + 6]
    body = "\n".join(" " + l for l in ctx)
    add = rng.choice(['    if not args:\n        raise ValueError("expected at least one argument")',
                      '    """Validated wrapper; see module docs."""',
                      '    if value is None:\n        return None'])
    adds = "\n".join("+" + l for l in add.splitlines())
    return (f"--- a/{path}\n+++ b/{path}\n@@ -{start + 1},6 +{start + 1},{6 + len(add.splitlines())} @@\n"
            f"{body.splitlines()[0]}\n{adds}\n" + "\n".join(body.splitlines()[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--shared-tokens", type=int, default=20_000, help="global prefix shared by ALL sessions")
    ap.add_argument("--session-tokens", type=int, default=8_000, help="per-session repo context")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id for exact token counts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    count = make_counter(a.tokenizer)
    corpus = load_corpus(a.seed)

    header = AGENT_INSTRUCTIONS + "\n# Tools\n" + tool_schemas() + "\n# Repository overview\n"
    shared_files, idx = take_files(corpus, 0, a.shared_tokens - count(header), count)
    shared_prefix = header + "".join(b for _, _, b in shared_files)

    with open(a.out, "w") as out:
        for s in range(a.sessions):
            sess_files, idx = take_files(corpus, idx, a.session_tokens, count)
            if not sess_files:
                raise SystemExit("ran out of corpus files; lower --sessions or token budgets")
            context = "Files currently open in the editor:\n" + "".join(b for _, _, b in sess_files)
            turns = []
            for t in range(a.turns):
                path, txt, _ = rng.choice(sess_files)
                funcs = re.findall(r"^def (\w+)\(", txt, re.M) or ["main"]
                func = rng.choice(funcs)
                turns.append({"user": f"In {path}, update `{func}` to {rng.choice(TASKS)}. Reply with a unified diff only.",
                              "canned_reply": canned_diff(path, txt, func, rng)})
            out.write(json.dumps({"id": f"agentic-{s}",
                                  "messages": [{"role": "system", "content": shared_prefix},
                                               {"role": "user", "content": context},
                                               {"role": "assistant", "content": "Understood. What should I change?"}],
                                  "turns": turns}) + "\n")
    print(f"wrote {a.sessions} sessions x {a.turns} turns to {a.out}; "
          f"shared prefix ~{count(shared_prefix)} tok, per-session context ~{a.session_tokens} tok")


if __name__ == "__main__":
    main()
