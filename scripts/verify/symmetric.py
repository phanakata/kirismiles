"""Mechanisms invariant under the two mirror symmetries of a Cho alpha^N block,
at a symmetric, slightly opened configuration."""
import numpy as np
from fractal import segments, covered, G
from generic import residual_jac

def linkage_full(word):
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
    members = {}
    for i in range(S):
        for j in range(S):
            members.setdefault(bid[find((i, j))], []).append((i, j))
    # body reference point: centroid of its squares (maps to the image body's centroid)
    C = np.array([np.mean([(i + .5, j + .5) for i, j in members[k]], axis=0) for k in range(len(roots))])
    sq_body = {sq: k for k, sqs in members.items() for sq in sqs}
    anchors = []
    for a, b, p in pins:
        ia, ib = sq_body[a], sq_body[b]
        if ia != ib:
            p = np.array(p, float); anchors.append((ia, ib, p - C[ia], p - C[ib]))
    return S, len(roots), C, anchors, sq_body

def mirror_ops(S, nb, sq_body):
    """3nb x 3nb matrices of the two mirrors acting on (x, y, theta) of all bodies."""
    ops = []
    for axis in (0, 1):
        M = np.zeros((3 * nb, 3 * nb))
        for (i, j), b in sq_body.items():
            ii, jj = (S - 1 - i, j) if axis == 0 else (i, S - 1 - j)
            b2 = sq_body[(ii, jj)]
            sx, sy = (-1, 1) if axis == 0 else (1, -1)
            M[2 * b2, 2 * b] = sx; M[2 * b2 + 1, 2 * b + 1] = sy
            M[2 * nb + b2, 2 * nb + b] = -1
        ops.append(M)
    return ops

def symmetric_F(word, eps=0.05, seed=0):
    S, nb, C, anchors, sq_body = linkage_full(word)
    Mx, My = mirror_ops(S, nb, sq_body)
    P = (np.eye(3 * nb) + Mx) @ (np.eye(3 * nb) + My) / 4     # projector on invariant velocities
    q0 = np.concatenate([C.ravel(), np.zeros(nb)])
    rng = np.random.default_rng(seed)
    q = q0 + eps * (P @ rng.standard_normal(3 * nb))
    for _ in range(200):
        r, J = residual_jac(q, nb, anchors)
        if np.linalg.norm(r) < 1e-13:
            break
        q -= np.linalg.lstsq(J, r, rcond=None)[0]
    r, J = residual_jac(q, nb, anchors)
    # invariant subspace basis
    w, V = np.linalg.eigh((P + P.T) / 2); B = V[:, w > 0.5]
    sv = np.linalg.svd(J @ B, compute_uv=False)
    null = B.shape[1] - int((sv > 1e-7 * sv[0]).sum())
    return null, np.linalg.norm(r)       # symmetric rigid motions: none (translations/rotation break a mirror)

if __name__ == "__main__":
    for w in ("a", "aa", "ab", "aaa", "aba", "aaaa"):
        print(f"level {len(w)} {w:5s}: symmetric F =", [symmetric_F(w, seed=s)[0] for s in range(3)])
