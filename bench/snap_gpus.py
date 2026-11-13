#!/usr/bin/env python3
"""Poll Lambda Cloud for capacity and auto-launch better GPUs the moment they appear.

SPENDS MONEY when run with --launch. Without --launch it only prints what it WOULD launch (dry run).
Guards: per-shape price cap, total $/hr cap across everything this script launches, max number of launches.
After a launch it waits for the IP, copies the kit over and starts setup_node.sh in tmux (unless --no-setup).
It never terminates anything: you terminate in the dashboard (or `--terminate <id>`).

  read -s LAMBDA_API_KEY; export LAMBDA_API_KEY
  python3 snap_gpus.py                      # dry run: shows matches
  python3 snap_gpus.py --launch --max-total-per-hour 60 --max-launches 2

Stdlib only, so it runs on your laptop without installing anything.
"""
import argparse, base64, json, os, re, socket, subprocess, sys, time, urllib.error, urllib.request

# Force IPv4: on networks with broken IPv6, urllib hangs on the AAAA address (curl falls back; urllib doesn't).
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **k: _orig_getaddrinfo(host, port, socket.AF_INET, *a, **k)

API = os.environ.get("LAMBDA_API_BASE", "https://cloud.lambdalabs.com/api/v1")
KIT = os.path.dirname(os.path.abspath(__file__))

# (regex on instance type name, max $/hr). Order = preference. Edit to taste.
DEFAULT_WANT = [
    (r"8x_b200", 60.0),
    (r"8x_h200", 40.0),
    (r"8x_h100", 32.0),
    (r"[124]x_b200", 30.0),
    (r"[124]x_h200", 20.0),
    (r"[124]x_h100", 18.0),
    (r"1x_gh200", 3.0),   # GH200 = 1 H100-class GPU with 96 GB HBM; fits NVFP4 with more KV room than H100 80GB
]


def call(method, path, key, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{key}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    # Lambda's Cloudflare front blocks the default "Python-urllib" user agent (HTTP 403, error code 1010).
    req.add_header("User-Agent", "lambda-bench-snap/1.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": repr(e)}


def available(key):
    d = call("GET", "/instance-types", key)
    if "error" in d:
        print("  api error:", d["error"], file=sys.stderr)
        return []
    out = []
    for name, v in d.get("data", {}).items():
        regions = [r["name"] for r in v.get("regions_with_capacity_available", [])]
        if regions:
            out.append((name, v["instance_type"]["price_cents_per_hour"] / 100, regions))
    return out


def setup_remote(ip, ssh_key):
    ssh = ["-i", ssh_key, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    for _ in range(60):  # sshd can lag the IP by a minute or two
        if subprocess.run(["ssh", *ssh, f"ubuntu@{ip}", "true"], capture_output=True).returncode == 0:
            break
        time.sleep(10)
    subprocess.run(["scp", *ssh, "-r", "-q", KIT, f"ubuntu@{ip}:~/"], check=True)
    subprocess.run(["ssh", *ssh, f"ubuntu@{ip}",
                    "tmux new -d -s main 'cd lambda-bench && ./setup_node.sh 2>&1 | tee setup.log; bash'"], check=True)
    print(f"  setup started on {ip}:  ssh {' '.join(ssh[:2])} ubuntu@{ip} -t 'tmux attach -t main'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="actually launch (spends money). Default: dry run")
    ap.add_argument("--max-total-per-hour", type=float, default=60.0, help="cap on summed $/hr of everything launched by this script")
    ap.add_argument("--max-launches", type=int, default=2)
    ap.add_argument("--ssh-key-name", default="lambda-bench", help="name of the SSH key as registered in Lambda")
    ap.add_argument("--ssh-key-file", default=os.path.expanduser("~/.ssh/lambda_ed25519"))
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--no-setup", action="store_true")
    ap.add_argument("--only", default=None, help="regex: only consider instance types matching this (e.g. '1x_b200|2x_b200')")
    ap.add_argument("--terminate", nargs="+", metavar="INSTANCE_ID", help="terminate these instances and exit")
    a = ap.parse_args()
    key = os.environ.get("LAMBDA_API_KEY") or sys.exit("export LAMBDA_API_KEY first")

    if a.terminate:
        print(call("POST", "/instance-operations/terminate", key, {"instance_ids": a.terminate}))
        return

    # Resolve the SSH key's name in Lambda by matching our local public key (names are whatever you typed in the UI).
    keys = call("GET", "/ssh-keys", key)
    if "error" in keys:
        sys.exit(f"could not list SSH keys: {keys['error']}")
    names = {k["name"]: k.get("public_key", "").split()[:2] for k in keys.get("data", [])}
    pub_path = a.ssh_key_file + ".pub"
    local = open(pub_path).read().split()[:2] if os.path.exists(pub_path) else None
    match = [n for n, pk in names.items() if pk == local]
    if match:
        a.ssh_key_name = match[0]
    elif a.ssh_key_name not in names:
        sys.exit(f"SSH key '{a.ssh_key_name}' not in Lambda and no key matches {pub_path}. "
                 f"Keys in your account: {list(names) or 'none'}. Add {pub_path} in the dashboard or pass --ssh-key-name.")
    print(f"using Lambda SSH key '{a.ssh_key_name}'")

    spent_per_hour, launched = 0.0, []
    print(f"{'LAUNCH' if a.launch else 'DRY RUN'}: caps ${a.max_total_per_hour}/hr total, {a.max_launches} launches")
    while len(launched) < a.max_launches:
        avail = available(key)
        pick = None
        for pat, cap in DEFAULT_WANT:
            for name, price, regions in avail:
                if a.only and not re.search(a.only, name):
                    continue
                if re.search(pat, name) and price <= cap and spent_per_hour + price <= a.max_total_per_hour:
                    pick = (name, price, regions[0]); break
            if pick:
                break
        stamp = time.strftime("%H:%M:%S")
        if not pick:
            print(f"{stamp} nothing wanted available ({len(avail)} shapes have capacity: "
                  f"{', '.join(n for n, _, _ in avail) or '-'})")
            time.sleep(a.interval); continue
        name, price, region = pick
        if not a.launch:
            print(f"{stamp} WOULD LAUNCH {name} in {region} at ${price:.2f}/hr (rerun with --launch)")
            time.sleep(a.interval); continue
        print(f"{stamp} LAUNCHING {name} in {region} at ${price:.2f}/hr")
        r = call("POST", "/instance-operations/launch", key, {"region_name": region, "instance_type_name": name,
                 "ssh_key_names": [a.ssh_key_name], "quantity": 1, "name": f"bench-{name}"})
        ids = (r.get("data") or {}).get("instance_ids")
        if not ids:
            err = str(r.get("error", r))
            if "capacity" not in err.lower() and "insufficient" not in err.lower():
                sys.exit(f"  launch failed with a non-capacity error, stopping: {err}")
            print("  launch failed on capacity (it often vanishes within seconds); retrying:", err)
            time.sleep(3); continue
        iid = ids[0]
        spent_per_hour += price
        launched.append((iid, name, price))
        print(f"  launched {iid}. Running total for this script: ${spent_per_hour:.2f}/hr. "
              f"Terminate later with: python3 snap_gpus.py --terminate {iid}")
        ip = None
        for _ in range(90):
            ip = ((call("GET", f"/instances/{iid}", key).get("data") or {}).get("ip"))
            if ip:
                break
            time.sleep(10)
        print(f"  {iid} ip={ip}")
        if ip and not a.no_setup:
            try:
                setup_remote(ip, a.ssh_key_file)
            except subprocess.CalledProcessError as e:
                print("  remote setup failed; do it by hand:", e)
    print("done. launched:", launched)


if __name__ == "__main__":
    main()
