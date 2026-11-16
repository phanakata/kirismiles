"""DOF of an isolated Cho block at a generic (slightly opened) configuration:
perturb the closed state, Gauss-Newton back onto the hinge constraints, then
count 3n - rank(J) - 3 there."""
import numpy as np
from fractal import segments, covered, G

def linkage(word):
    S, segs = segments(word); S = int(S)
    parent = {(i, j): (i, j) for i in range(S) for j in range(S)}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pins = []
    for i in range(S):
        for j in range(S):
            for (line, c, t0, a, b) in (("v", i + 1, j, (i, j), (i + 1, j)),
                                        ("h", j + 1, i, (i, j), (i, j + 1))):
                if b[0] >= S or b[1] >= S:
                    continue
                if not covered(segs, line, c, t0 + 0.5):
                    parent[find(a)] = find(b); continue
                for t, end in ((t0 + G / 2, t0), (t0 + 1 - G / 2, t0 + 1)):
                    if not covered(segs, line, c, t):
                        pins.append((a, b, (c, end) if line == "v" else (end, c)))
    roots = sorted({find((i, j)) for i in range(S) for j in range(S)})
    bid = {r: k for k, r in enumerate(roots)}
    H = []
    for a, b, p in pins:
        ia, ib = bid[find(a)], bid[find(b)]
        if ia != ib:
            H.append((ia, ib, np.array(p, float)))
    # body frames: origin at the root square centre, angle 0 in the closed state
    C = np.array([(r[0] + .5, r[1] + .5) for r in roots])
    anchors = [(ia, ib, p - C[ia], p - C[ib]) for ia, ib, p in H]
    return len(roots), C, anchors

def R(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s], [s, c]])

def residual_jac(q, nb, anchors):
    X = q[:2 * nb].reshape(nb, 2); T = q[2 * nb:]
    res, rows = [], []
    for ia, ib, ra, rb in anchors:
        pa = X[ia] + R(T[ia]) @ ra; pb = X[ib] + R(T[ib]) @ rb
        res.extend(pa - pb)
        dRa = R(T[ia] + np.pi / 2) @ ra; dRb = R(T[ib] + np.pi / 2) @ rb
        for k in (0, 1):
            r = np.zeros(3 * nb)
            r[2 * ia + k] += 1; r[2 * ib + k] -= 1
            r[2 * nb + ia] += dRa[k]; r[2 * nb + ib] -= dRb[k]
            rows.append(r)
    return np.array(res), np.array(rows)

def generic_dof(word, eps=0.08, seed=0):
    nb, C, anchors = linkage(word)
    rng = np.random.default_rng(seed)
    q = np.concatenate([C.ravel(), np.zeros(nb)]) + eps * rng.standard_normal(3 * nb)
    for _ in range(200):
        r, J = residual_jac(q, nb, anchors)
        if np.linalg.norm(r) < 1e-12:
            break
        q -= np.linalg.lstsq(J, r, rcond=None)[0]
    r, J = residual_jac(q, nb, anchors)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-7 * sv[0]).sum())
    return 3 * nb - rank - 3, np.linalg.norm(r)

if __name__ == "__main__":
    for w in ("a", "aa", "ab", "ba", "aaa", "aba", "aaaa", "baba"):
        Fs = [generic_dof(w, seed=s) for s in range(3)]
        print(f"level {len(w)} {w:5s}: F = {[f for f, _ in Fs]}  residual {max(r for _, r in Fs):.1e}")
