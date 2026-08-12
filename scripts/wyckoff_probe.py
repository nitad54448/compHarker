# wyckoff_probe.py
#
# Standalone check that this cctbx build gives everything the Wyckoff search
# needs, for ONE space group, before regenerating all 230.
#
#   python wyckoff_probe.py 62          # Pnma
#   python wyckoff_probe.py 206         # Ia-3
#   python wyckoff_probe.py "P 21/c"    # symbols work too
#
# Nothing here is imported by the generator; it is a diagnostic. Once it prints
# a clean table, paste the three functions from wyckoff_patch.py into
# cctbx_generate_sg_harker_v6.py and run that.

import sys
import math
from fractions import Fraction

try:
    from cctbx import sgtbx
except ImportError:
    print("cctbx is not importable in this environment.\n")
    print("Create one with miniforge:\n")
    print("    conda create -n cctbx -c conda-forge cctbx-base python=3.11")
    print("    conda activate cctbx")
    print("    python wyckoff_probe.py 62\n")
    print("Note cctbx is NOT on PyPI in a usable form - pip install cctbx will")
    print("not give you a working build. It has to come from conda-forge.")
    sys.exit(1)


# --- Wyckoff table accessor probing -----------------------------------------
# cctbx does not put the Wyckoff table at one fixed place; which route works
# depends on the build. This is the same probe the generator uses, reproduced
# here so the script stands alone.

def get_wyckoff_table_object(sg_info):
    def via_info(i):
        return i.wyckoff_table()

    def via_submodule(i):
        from cctbx.sgtbx import wyckoff as w
        return w.table(i)

    def via_import(i):
        import importlib
        w = importlib.import_module("cctbx.sgtbx.wyckoff")
        return w.table(i)

    def via_attr(i):
        return sgtbx.wyckoff.table(i)

    def via_flat(i):
        return sgtbx.wyckoff_table(i)

    routes = [
        ("space_group_info.wyckoff_table()", via_info),
        ("from cctbx.sgtbx import wyckoff", via_submodule),
        ("importlib cctbx.sgtbx.wyckoff", via_import),
        ("sgtbx.wyckoff.table()", via_attr),
        ("sgtbx.wyckoff_table()", via_flat),
    ]
    errors = []
    for name, fn in routes:
        try:
            t = fn(sg_info)
            if t is not None:
                return t, name, errors
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return None, None, errors


# --- the new fields ---------------------------------------------------------

def rank3(num, den):
    """Exact rank of a 3x3 matrix given as 9 numerators over one denominator."""
    rows = [[Fraction(int(num[r * 3 + c]), int(den)) for c in range(3)] for r in range(3)]
    rank = 0
    for col in range(3):
        pivot = next((r for r in range(rank, 3) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pv = rows[rank][col]
        rows[rank] = [v / pv for v in rows[rank]]
        for r in range(3):
            if r != rank and rows[r][col] != 0:
                f = rows[r][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


# No rational relationship between the coordinates and no small denominators,
# so projecting it lands on a GENERIC point of the position rather than
# accidentally on a higher-symmetry sub-position.
PROBE = (0.1357913, 0.2468135, 0.3791357)


def apply_op(op, p):
    r = op.r().as_double()
    t = op.t().as_double()
    return (r[0]*p[0] + r[1]*p[1] + r[2]*p[2] + t[0],
            r[3]*p[0] + r[4]*p[1] + r[5]*p[2] + t[1],
            r[6]*p[0] + r[7]*p[1] + r[8]*p[2] + t[2])


def coset_ops(sg, special_op):
    """Operator indices giving DISTINCT images of a site on this position."""
    p = apply_op(special_op, PROBE)
    seen, ops = [], []
    for i in range(sg.order_z()):
        n = apply_op(sg(i), p)
        n = tuple(v - math.floor(v) for v in n)
        dup = False
        for s in seen:
            d = [n[k] - s[k] for k in range(3)]
            d = [v - round(v) for v in d]
            if all(abs(v) < 1e-6 for v in d):
                dup = True
                break
        if not dup:
            seen.append(n)
            ops.append(i)
    return ops


def main():
    if len(sys.argv) < 2:
        print("usage: python wyckoff_probe.py <space group number or symbol>")
        sys.exit(2)

    arg = sys.argv[1]
    try:
        sg_info = sgtbx.space_group_info(int(arg))
    except ValueError:
        sg_info = sgtbx.space_group_info(arg)

    sg = sg_info.group()
    print(f"space group : {sg_info.symbol_and_number()}")
    print(f"order_z     : {sg.order_z()}   (operators including centring)")
    print()

    table, route, errors = get_wyckoff_table_object(sg_info)
    if table is None:
        print("No Wyckoff accessor worked in this build. Routes tried:")
        for e in errors:
            print("   " + e)
        sys.exit(1)
    print(f"Wyckoff table via: {route}")
    print()

    hdr = f"{'pos':>6}  {'site sym':<10} {'special_op':<16} {'n_free':>6}  {'cosets':>6}  ok"
    print(hdr)
    print("-" * len(hdr))

    problems = 0
    for i in range(table.size()):
        pos = table.position(i)
        letter = pos.letter()
        mult = int(pos.multiplicity())
        try:
            site_sym = str(pos.point_group_type())
        except Exception:
            try:
                site_sym = str(pos.site_symmetry_group().point_group_type())
            except Exception:
                site_sym = "?"

        sop = pos.special_op()
        r, t = sop.r(), sop.t()
        p_num = [int(v) for v in r.num()]
        p_den = int(r.den())
        n_free = rank3(p_num, p_den)

        ops = coset_ops(sg, sop)
        ok = (len(ops) == mult)
        if not ok:
            problems += 1

        print(f"{str(mult) + letter:>6}  {site_sym:<10} {str(sop):<16} "
              f"{n_free:>6}  {len(ops):>6}  {'yes' if ok else 'NO  <-- report this'}")

    print()
    if problems:
        print(f"{problems} position(s) gave a coset count that disagrees with the")
        print("tabulated multiplicity. That means the probe point landed on a")
        print("higher-symmetry sub-position. Send me the table above.")
        sys.exit(1)

    print("All positions consistent: coset count matches multiplicity everywhere.")
    print()
    print("Sanity check for Pnma (62): 4a and 4b should show n_free 0, 4c should")
    print("show n_free 2 with special_op x,1/4,z, and 8d n_free 3 with 8 cosets.")


if __name__ == "__main__":
    main()
