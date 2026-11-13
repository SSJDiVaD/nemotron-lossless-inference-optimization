#!/usr/bin/env python3
"""Dedicated-endpoint lifecycle for the provider comparison: discover shapes, create, wait, describe, delete.

  endpoint.py discover --provider fireworks|together|all            # list hardware shapes for the model (no spend)
  endpoint.py create   --provider P --shape S --state-file F         # create 1-replica dedicated endpoint (SPENDS MONEY)
  endpoint.py wait     --state-file F [--timeout-min 60]            # until a real chat completion returns 200
  endpoint.py describe --state-file F                               # endpoint metadata -> stdout (JSON)
  endpoint.py delete   --state-file F                               # idempotent teardown

Providers:
  fireworks : REST API (https://api.fireworks.ai/v1/accounts/{account}/...). Env: FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID
  together  : official `tg` CLI (Together CLI >= 3.0). Env: TOGETHER_API_KEY (+ optional TOGETHER_PROJECT_ID)
  mock      : local mock server (bench/provider/mock_server.py) for free end-to-end tests

The state file (JSON) records everything needed to reach and tear down the endpoint, and is written BEFORE waiting,
so a crashed or cancelled run can always be cleaned up (the GitHub workflow's always() teardown job reads it).
Stdlib only.
"""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request

FW_API = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai/v1")  # overridable for tests
FW_INFER = "https://api.fireworks.ai/inference/v1"
FW_MODELS = ["accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4"]
TG_INFER = "https://api.together.xyz/v1"
TG_MODEL_PATTERN = r"(?i)nemotron.*3\.?5.*super"
UA = {"User-Agent": "nemotron-provider-bench/1.0"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def http(method, url, key, body=None, timeout=60):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", **UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:2000].decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"error": repr(e)}


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var {name}")
    return v


def run(cmd, check=True):
    log("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        log(r.stdout.strip()[-4000:])
    if r.stderr.strip():
        log(r.stderr.strip()[-2000:])
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


# ----------------------------------------------------------------------------------------------- Fireworks (REST)
def fw_discover():
    key, acct = env("FIREWORKS_API_KEY"), env("FIREWORKS_ACCOUNT_ID")
    out = []
    for model in FW_MODELS:
        code, d = http("POST", f"{FW_API}/accounts/{acct}/deploymentShapeVersions:match", key,
                       {"createDeploymentRequest": {"deployment": {"baseModel": model}}})
        if code != 200:
            log(f"fireworks match failed for {model}: HTTP {code} {d}")
            continue
        for v in d.get("deploymentShapeVersions", []):
            snap = v.get("snapshot", {})
            out.append({"provider": "fireworks", "model": model, "shape": v.get("name"),
                        "gpu": snap.get("acceleratorType"), "count": snap.get("acceleratorCount"),
                        "precision": snap.get("precision"), "validated": v.get("validated")})
    return out


def fw_create(shape, model, state_file):
    key, acct = env("FIREWORKS_API_KEY"), env("FIREWORKS_ACCOUNT_ID")
    dep_id = f"provider-bench-{int(time.time())}"
    name = f"accounts/{acct}/deployments/{dep_id}"
    st = {"provider": "fireworks", "deployment": name, "base_url": FW_INFER, "model": name,
          "api_key_env": "FIREWORKS_API_KEY", "shape": shape, "pending": True}
    save(st, state_file)  # written BEFORE the API call: teardown works even if the create response is lost
    # One GPU max. min=0 + 30-min scale-to-zero is a provider-side safety net: if every teardown path fails,
    # Fireworks releases the GPU after 30 idle minutes (benchmark traffic keeps it busy until then).
    body = {"baseModel": model, "deploymentShape": shape, "minReplicaCount": 0, "maxReplicaCount": 1,
            "autoscalingPolicy": {"scaleToZeroWindow": "1800s"}, "displayName": "provider-bench (auto-deleted)"}
    code, d = http("POST", f"{FW_API}/accounts/{acct}/deployments?deploymentId={dep_id}", key, body)
    if code != 200:
        log(f"fireworks create failed: HTTP {code} {d}; state kept so teardown can verify/delete")
        sys.exit(1)
    st.update(deployment=d.get("name") or name, model=d.get("name") or name, pending=False, create_response=d)
    return st


def fw_describe(st):
    code, d = http("GET", f"{FW_API}/{st['deployment']}", env("FIREWORKS_API_KEY"))
    return {"http": code, **d}


def fw_delete(st):
    code, d = http("DELETE", f"{FW_API}/{st['deployment']}", env("FIREWORKS_API_KEY"))
    if code in (200, 204, 404):  # 404 = already deleted
        return True
    log(f"fireworks REST delete HTTP {code} {d}; trying firectl")
    if shutil.which("firectl"):
        return run(["firectl", "deployment", "delete", st["deployment"].split("/")[-1]], check=False).returncode == 0
    return False


# --------------------------------------------------------------------------------------------- Together (tg CLI)
def tg(args, check=True):
    if not shutil.which("tg"):
        sys.exit("Together CLI `tg` not found (install the Together CLI >= 3.0)")
    r = run(["tg", "--json", *args], check=check)
    try:
        return r.returncode, json.loads(r.stdout) if r.stdout.strip() else {}
    except json.JSONDecodeError:
        return r.returncode, {"raw": r.stdout}


def tg_discover():
    out = []
    _, pub = tg(["models", "public"], check=False)
    items = pub if isinstance(pub, list) else pub.get("data") or pub.get("models") or []
    names = sorted({(m.get("modelName") or m.get("id") or m.get("name") or "") for m in items if isinstance(m, dict)})
    hits = [n for n in names if re.search(TG_MODEL_PATTERN, n)]
    if not hits:
        log("together: no public model matched", TG_MODEL_PATTERN, "- raw listing above")
    for n in hits:
        _, cfg = tg(["models", "configs", n], check=False)
        cfgs = cfg if isinstance(cfg, list) else cfg.get("data") or cfg.get("configs") or [cfg]
        for c in cfgs:
            if not isinstance(c, dict):
                continue
            out.append({"provider": "together", "model": n, "shape": c.get("id") or c.get("name"),
                        "gpu": c.get("hardware") or c.get("gpuType") or c.get("gpu"),
                        "count": c.get("gpuCount") or c.get("numGpus"), "precision": c.get("precision") or c.get("quantization"),
                        "raw": c})
    return out


def tg_create(shape, model, state_file):
    name = f"provider-bench-{int(time.time())}"
    save({"provider": "together", "endpoint": name, "endpoint_id": name, "base_url": TG_INFER, "model": name,
          "api_key_env": "TOGETHER_API_KEY", "shape": shape, "pending": True}, state_file)  # before the API call
    args = ["endpoints", "deploy", model, "--endpoint", name, "--min-replicas", "1", "--max-replicas", "1"]
    if shape and shape.startswith("cr_"):
        args += ["--config", shape]
    code, d = tg(args)
    ep_name = d.get("name") or (d.get("endpoint") or {}).get("name") or name
    ep_id = d.get("id") or (d.get("endpoint") or {}).get("id") or name
    return {"provider": "together", "endpoint": ep_name, "endpoint_id": ep_id, "base_url": TG_INFER, "model": ep_name,
            "api_key_env": "TOGETHER_API_KEY", "shape": shape, "create_response": d}


def tg_describe(st):
    _, d = tg(["endpoints", "get", st["endpoint_id"]], check=False)
    return d


def tg_delete(st):
    for ref in (st["endpoint_id"], st["endpoint"]):
        if tg(["endpoints", "rm", ref, "--force"], check=False)[0] == 0:
            return True
    # already gone (e.g. deleted by the runner before this independent teardown) counts as success
    return tg(["endpoints", "get", st["endpoint_id"]], check=False)[0] != 0


# ---------------------------------------------------------------------------------------------------- mock
def mock_create(shape, model, state_file=None):
    return {"provider": "mock", "base_url": os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:18765/v1"),
            "model": "mock/provider-bench-endpoint", "api_key_env": None, "shape": shape}


# ------------------------------------------------------------------------------------------------- common
def save(st, path):
    st.setdefault("created_at", time.time())
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)


def retry(fn, st, tries=4):
    for i in range(tries):
        try:
            if fn(st):
                return True
        except SystemExit as e:
            log(f"delete attempt {i + 1} error: {e}")
        time.sleep(float(os.environ.get("RETRY_SLEEP_S", "15")) * (i + 1))
    return False


# ------------------------------------------------------------------------------------------- janitor
def fw_list():
    key, acct = env("FIREWORKS_API_KEY"), env("FIREWORKS_ACCOUNT_ID")
    out, token = [], ""
    while True:
        code, d = http("GET", f"{FW_API}/accounts/{acct}/deployments?pageSize=200" + (f"&pageToken={token}" if token else ""), key)
        if code != 200:
            log(f"fireworks list failed: HTTP {code} {d}"); return out
        out += d.get("deployments", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def janitor(max_age_h, dry):
    """Delete provider-bench-* endpoints older than max_age_h at every provider we have credentials for."""
    now, n = time.time(), 0
    if os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_ACCOUNT_ID"):
        for dep in fw_list():
            name = dep.get("name", "")
            m = re.search(r"provider-bench-(\d{9,})", name)
            if not m or dep.get("state") in ("DELETED", "DELETING"):
                continue
            age_h = (now - int(m.group(1))) / 3600
            log(f"fireworks {name} state={dep.get('state')} age={age_h:.1f}h")
            if age_h > max_age_h:
                n += 1
                if not dry:
                    retry(fw_delete, {"deployment": name})
    if os.environ.get("TOGETHER_API_KEY") and shutil.which("tg"):
        _, lst = tg(["endpoints", "list"], check=False)
        items = lst if isinstance(lst, list) else lst.get("data") or lst.get("endpoints") or []
        for ep in items:
            nm = str(ep.get("name", "")) if isinstance(ep, dict) else ""
            m = re.search(r"provider-bench-(\d{9,})", nm)
            if not m:
                continue
            age_h = (now - int(m.group(1))) / 3600
            log(f"together {nm} age={age_h:.1f}h")
            if age_h > max_age_h:
                n += 1
                if not dry:
                    retry(tg_delete, {"endpoint": nm, "endpoint_id": ep.get("id") or nm})
    log(f"janitor: {n} stale provider-bench endpoint(s) {'would be ' if dry else ''}deleted")


def chat_ok(st):
    key = os.environ.get(st["api_key_env"]) if st.get("api_key_env") else "none"
    code, d = http("POST", st["base_url"].rstrip("/") + "/chat/completions", key,
                   {"model": st["model"], "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]}, timeout=120)
    return code, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "create", "wait", "describe", "delete", "janitor"])
    ap.add_argument("--max-age-hours", type=float, default=6)
    ap.add_argument("--provider", choices=["fireworks", "together", "mock", "all"], default="all")
    ap.add_argument("--shape", default=None, help="Fireworks shape-version name or Together config id (cr_...)")
    ap.add_argument("--model", default=None, help="override base model id")
    ap.add_argument("--state-file", default="endpoint_state.json")
    ap.add_argument("--timeout-min", type=float, default=60)
    ap.add_argument("--dry-run", action="store_true", help="create: print what would be created, spend nothing")
    a = ap.parse_args()

    if a.cmd == "discover":
        rows = []
        for p, fn in (("fireworks", fw_discover), ("together", tg_discover)):
            if a.provider in (p, "all"):
                try:
                    rows += fn()
                except SystemExit as e:
                    log(f"{p}: skipped ({e})")
        print(json.dumps(rows, indent=1, default=str))
        hopper = [r for r in rows if re.search(r"(?i)h100|h200", str(r.get("gpu")))]
        log(f"\n{len(rows)} shapes found; Hopper (H100/H200) shapes: {len(hopper)}")
        for r in rows:
            log(f"  {r['provider']:9s} {str(r.get('gpu')):22s} x{r.get('count')}  {str(r.get('precision')):10s} {r.get('shape')}")
        return

    if a.cmd == "create":
        if a.provider not in ("fireworks", "together", "mock"):
            sys.exit("create needs --provider fireworks|together|mock")
        if a.provider != "mock" and not a.shape:
            sys.exit("create needs an explicit --shape (pick one from `discover`)")
        model = a.model or {"fireworks": FW_MODELS[0], "together": None, "mock": "mock"}[a.provider]
        if a.provider == "together" and not model:
            sys.exit("together create needs --model (the modelName shown by `discover`)")
        if a.dry_run:
            print(json.dumps({"would_create": {"provider": a.provider, "model": model, "shape": a.shape,
                                               "replicas": "min=max=1"}}))
            return
        st = {"fireworks": fw_create, "together": tg_create, "mock": mock_create}[a.provider](a.shape, model, a.state_file)
        save(st, a.state_file)
        print(json.dumps({k: st[k] for k in ("provider", "base_url", "model")}))
        return

    if a.cmd == "janitor":
        janitor(a.max_age_hours, a.dry_run); return

    if not os.path.exists(a.state_file):
        if a.cmd == "delete":
            log("no state file: nothing to delete"); return
        sys.exit(f"state file {a.state_file} not found")
    st = json.load(open(a.state_file))

    if a.cmd == "wait":
        deadline = time.time() + a.timeout_min * 60
        while time.time() < deadline:
            code, d = chat_ok(st)
            if code == 200:
                log(f"endpoint ready after {(time.time() - st.get('created_at', time.time())) / 60:.1f} min; model={d.get('model')}")
                print(json.dumps({"ready": True, "served_model": d.get("model")}))
                return
            log(f"not ready yet: HTTP {code} {str(d)[:200]}")
            time.sleep(30)
        sys.exit(f"endpoint not ready after {a.timeout_min} min")

    if a.cmd == "describe":
        d = {"fireworks": fw_describe, "together": tg_describe, "mock": lambda s: {"mock": True}}[st["provider"]](st)
        print(json.dumps({"state": st, "describe": d}, indent=1, default=str))
        return

    if a.cmd == "delete":
        ok = retry({"fireworks": fw_delete, "together": tg_delete, "mock": lambda s: True}[st["provider"]], st)
        log("deleted" if ok else "DELETE FAILED - check the provider console now")
        if ok:
            os.replace(a.state_file, a.state_file + ".deleted")
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
