"""Single cut: linear buckling over the two shape parameters b/W and L/W.

In the FvK plate the buckling displacement scales exactly as h^2, so by
dimensional analysis Delta_c = (h^2 / W) G(b/W, L/W): the thickness and the
overall size carry no information. This sweep maps G over b/W and L/W at
W = 100 mm, plus a rescaled sheet (W = 60 mm) as a numerical check that G does
not depend on the size. Runs one case at a time and appends each row to the CSV
as soon as it is done, so an interrupted sweep keeps its rows; rows already in
the file are skipped.

    ./run.sh python3 -u scripts/single_cut_shape_sweep.py
Writes results/single_cut/shape_sweep.csv
"""
import csv
import gc
import os
import time


from kirigami.fvk import Material, Sheet
from kirigami.geometry import single_cut

OUT = "results/single_cut/shape_sweep.csv"
E, NU, H = 3500.0, 0.38, 0.127
BW = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LW = [0.6, 1.0, 1.4, 1.82, 2.5, 3.5]
CASES = [(100.0, bw, lw) for lw in LW for bw in BW] + [(60.0, 0.5, 1.82), (60.0, 0.8, 1.82)]
COLS = ["W", "L", "b", "h", "b_over_W", "L_over_W", "F_unit", "Delta_c", "Delta_c2",
        "G", "Dc_b2_over_h2W", "n_cells", "seconds"]


def done():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(float(r["W"]), float(r["b_over_W"]), float(r["L_over_W"])) for r in csv.DictReader(f)}


def main():
    have = done()
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(COLS)
        for W, bw, lw in CASES:
            if (W, bw, lw) in have:
                continue
            t0 = time.time()
            L, b = round(lw * W, 3), round(bw * W, 3)
            msh = single_cut(L, W, b, h_far=4.0, h_cut=0.35)
            sh = Sheet(msh, L, Material(E, NU, H))
            _, N1, F1 = sh.solve_inplane(1.0)
            lams, _ = sh.linear_buckling(N1, nev=2)
            Dc = lams[0]
            row = [W, L, b, H, bw, lw, F1, Dc, lams[1] if len(lams) > 1 else "",
                   Dc * W / H**2, Dc * b**2 / (H**2 * W), msh.topology.index_map(2).size_local,
                   round(time.time() - t0, 1)]
            wr.writerow(row)
            f.flush()
            print(f"W={W:g} b/W={bw} L/W={lw}: Dc={Dc:.6f} mm  Dc b^2/(h^2 W)={row[10]:.3f}  "
                  f"({row[-1]}s, {row[11]} cells)", flush=True)
            del msh, sh, N1
            gc.collect()


if __name__ == "__main__":
    main()
