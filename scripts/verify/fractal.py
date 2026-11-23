"""Cho et al. (PNAS 2014) fractal cuts: build the cut segments of a level-N
pattern from the alpha/beta motifs of their Fig. 2, and count the rigid-unit
mechanisms of the periodic sheet (rigid finest squares, pin hinges, cell allowed
to strain) to compare with F_N = 1, 2, 6.

Motif on a square (x, y, s), centre (X, Y):
  alpha: a vertical cut through the centre stopping g short of the top and bottom
         edges, and horizontal arms from each side edge to g short of the centre;
  beta:  the same rotated by 90 degrees.
"""
import numpy as np

G = 1e-3          # hinge gap (rigid-unit limit: hinges are points)


def segments(word, N0=1):
    """Cut segments ((x0,y0),(x1,y1)) of one level-0 square of side 2**len(word)."""
    S = 2 ** len(word)
    segs = []

    def motif(x, y, s, m):
        X, Y = x + s / 2, y + s / 2
        if m == "a":
            segs.append(((X, y + G), (X, y + s - G)))
            segs.append(((x, Y), (X - G, Y)))
            segs.append(((X + G, Y), (x + s, Y)))
        else:
            segs.append(((x + G, Y), (x + s - G, Y)))
            segs.append(((X, y), (X, Y - G)))
            segs.append(((X, Y + G), (X, y + s)))

    def rec(x, y, s, level):
        if level == len(word):
            return
        motif(x, y, s, word[level])
        h = s / 2
        for dx in (0, h):
            for dy in (0, h):
                rec(x + dx, y + dy, h, level + 1)

    rec(0.0, 0.0, float(S), 0)
    return S, segs


def covered(segs, line, c, t):
    """Is point t on the line x=c ('v') or y=c ('h') covered by a cut?"""
    for (x0, y0), (x1, y1) in segs:
        if line == "v" and abs(x0 - c) < 1e-9 and abs(x1 - c) < 1e-9:
            if min(y0, y1) - 1e-12 <= t <= max(y0, y1) + 1e-12:
                return True
        if line == "h" and abs(y0 - c) < 1e-9 and abs(y1 - c) < 1e-9:
            if min(x0, x1) - 1e-12 <= t <= max(x0, x1) + 1e-12:
                return True
    return False


def bodies_and_hinges(word):
    S, segs = segments(word)
    S = int(S)
    # periodic: replicate segments of neighbouring cells so edges on the cell boundary see them
    segs = [((a[0] + i * S, a[1] + j * S), (b[0] + i * S, b[1] + j * S))
            for (a, b) in segs for i in (-1, 0, 1) for j in (-1, 0, 1)]
    parent = {(i, j): (i, j) for i in range(S) for j in range(S)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = []                                   # (sqA, sqB, point, wrap) hinges
    for i in range(S):
        for j in range(S):
            # edge with the right neighbour: line x = i+1, t in [j, j+1]
            for (line, c, t0, a, b, wrap) in (
                    ("v", i + 1, j, (i, j), ((i + 1) % S, j), ((i + 1) // S, 0)),
                    ("h", j + 1, i, (i, j), (i, (j + 1) % S), (0, (j + 1) // S))):
                mid = covered(segs, line, c, t0 + 0.5)
                if not mid:
                    links.append(("weld", a, b, None, wrap))
                    continue
                for t, end in ((t0 + G / 2, t0), (t0 + 1 - G / 2, t0 + 1)):
                    if not covered(segs, line, c, t):
                        p = (c, end) if line == "v" else (end, c)
                        links.append(("pin", a, b, p, wrap))
    for kind, a, b, p, wrap in links:
        if kind == "weld" and wrap == (0, 0):
            parent[find(a)] = find(b)
    return S, parent, find, links


def mechanisms(word):
    S, parent, find, links = bodies_and_hinges(word)
    roots = sorted({find((i, j)) for i in range(S) for j in range(S)})
    bid = {r: k for k, r in enumerate(roots)}
    nb = len(roots)
    # body reference point: centre of its root square; unknowns per body (u, v, w) + cell strain (exx, eyy, exy)
    ref = {r: (r[0] + .5, r[1] + .5) for r in roots}
    nv = 3 * nb + 3
    rows = []

    def add(a, b, p, wrap):
        ra, rb = find(a), find(b)
        ia, ib = bid[ra], bid[rb]
        pa = p
        pb = (p[0] - wrap[0] * S, p[1] - wrap[1] * S) if p is not None else None
        # position of the hinge seen from b's copy is shifted by the lattice vector;
        # displacement jump across the cell = E . (lattice vector)
        L = (wrap[0] * S, wrap[1] * S)
        for comp in (0, 1):
            r = np.zeros(nv)
            ca, cb = ref[ra], ref[rb]
            dxa, dya = pa[0] - ca[0], pa[1] - ca[1]
            dxb, dyb = pb[0] - cb[0], pb[1] - cb[1]
            r[3 * ia + comp] += 1
            r[3 * ib + comp] -= 1
            r[3 * ia + 2] += -dya if comp == 0 else dxa
            r[3 * ib + 2] -= -dyb if comp == 0 else dxb
            # a's copy sits one lattice vector further than b's: u_a(x) = u_b(x - L) + E L
            e = 3 * nb
            if comp == 0:
                r[e + 0] -= L[0]; r[e + 2] -= L[1]
            else:
                r[e + 1] -= L[1]; r[e + 2] -= L[0]
            rows.append(r)

    for kind, a, b, p, wrap in links:
        if kind == "pin":
            add(a, b, p, wrap)
        elif kind == "weld" and wrap != (0, 0):
            # welded across the cell boundary: two pins at the edge ends fix the bond
            i, j = a
            if wrap[0]:
                add(a, b, (i + 1, j), wrap); add(a, b, (i + 1, j + 1), wrap)
            else:
                add(a, b, (i, j + 1), wrap); add(a, b, (i + 1, j + 1), wrap)
    J = np.array(rows) if rows else np.zeros((0, nv))
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    return nv - rank - 2, nb          # minus the two rigid translations


if __name__ == "__main__":
    for w in ("a", "aa", "ab", "aaa", "aba"):
        F, nb = mechanisms(w)
        print(f"level {len(w)} motifs {w:4s}: mechanisms F = {F}  (rigid bodies {nb})")
