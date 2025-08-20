import os, numpy as np

root = r"C:\Users\thari\Desktop\Uni-Oulu\RA-CMVS\PC-detection\MM-Fi"
min_len = 30  # match your config

envs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d)) and d.upper().startswith("E")]
print("Envs:", envs)

stats = {"total_sequences":0, "valid_sequences":0, "too_short":0, "bad_shape":0}
per_env = {}

for e in envs:
    epath = os.path.join(root, e)
    subs = [s for s in os.listdir(epath) if os.path.isdir(os.path.join(epath, s)) and s.upper().startswith("S")]
    ok = 0; bad = 0
    for s in subs:
        spath = os.path.join(epath, s)
        acts = [a for a in os.listdir(spath) if os.path.isdir(os.path.join(spath, a)) and a.upper().startswith("A")]
        for a in acts:
            stats["total_sequences"] += 1
            g = os.path.join(spath, a, "ground_truth.npy")
            if not os.path.exists(g):
                bad += 1
                continue
            try:
                arr = np.load(g)
                if arr.ndim!=3 or arr.shape[1:]!=(17,3):
                    stats["bad_shape"] += 1; bad += 1; continue
                if arr.shape[0] < min_len:
                    stats["too_short"] += 1; bad += 1; continue
                ok += 1
            except Exception:
                bad += 1
    per_env[e] = {"valid": ok, "invalid": bad}

print("Per-env counts:", per_env)
print("Totals:", stats)