# wyckoff_patch.py
#
# Drop-in replacement for get_wyckoff_table() in cctbx_generate_sg_harker_v6.py,
# plus two helpers it needs. Copy these three functions over the existing
# get_wyckoff_table(); nothing else in the generator changes.
#
# WHY: the current table emits special_op as a STRING ("x,1/4,z"). A string is
# readable but not usable - the browser cannot project a trial coordinate onto
# it, cannot count its degrees of freedom, and cannot know which symmetry
# operators produce DISTINCT images of a site sitting on it. All three are
# needed to search Wyckoff space instead of coordinate space.
#
# ADDED PER POSITION:
#   P_num, P_den   rotation part of special_op, 9 ints / common denominator.
#                  x_on_position = P . q + T for ANY q. P is idempotent, so
#                  projecting twice is projecting once.
#   T_num, T_den   translation part, 3 ints / common denominator.
#   n_free         rank(P) - the true number of continuous parameters
#                  (0 for 4a, 2 for 4c (x,1/4,z), 3 for a general position).
#   coset_ops      indices into this setting's sym_ops list giving exactly
#                  `multiplicity` DISTINCT images. Depends only on the Wyckoff
#                  position, not on where along it the atom sits, so it can be
#                  precomputed once and the shader's distance-based
#                  special-position collapse deleted outright.

from collections import OrderedDict
from fractions import Fraction


def _rank3(num, den):
    """Rank of a 3x3 matrix given as 9 integer numerators over one denominator.

    Exact: Gaussian elimination over Fraction, so a matrix like the projector
    for (x, x, z) - which has a 1/2 in it - is never misjudged by rounding.
    The denominator scales the whole matrix and cannot change its rank, but it
    is carried anyway so the arithmetic reads as the matrix it represents.
    """
    rows = [[Fraction(int(num[r * 3 + c]), int(den)) for c in range(3)] for r in range(3)]
    rank = 0
    for col in range(3):
        pivot = None
        for r in range(rank, 3):
            if rows[r][col] != 0:
                pivot = r
                break
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


# A point with no rational relationship between its coordinates and no small
# denominators. Projecting it through special_op lands on a GENERIC point of
# the Wyckoff position rather than accidentally on a higher-symmetry
# sub-position, which would return too few coset operators.
_PROBE = (0.1357913, 0.2468135, 0.3791357)


def _apply(op, p):
    r = op.r().as_double()
    t = op.t().as_double()
    return (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
            r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
            r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2])


def _coset_ops(sg, special_op, multiplicity):
    """Operator indices producing distinct images of a site on this position.

    The stabiliser of a Wyckoff position is the same for every point on it, so
    the coset representatives are a property of the POSITION. Computed once
    here, they let a consumer generate exactly `multiplicity` atoms with no
    coincidence test and no tolerance to tune.

    Returns (ops, exact) where exact is False if the count disagrees with the
    tabulated multiplicity - which would mean the probe point was unlucky, and
    the consumer should fall back to distance-based collapse for this position.
    """
    p = _apply(special_op, _PROBE)
    seen = []
    ops = []
    for i in range(sg.order_z()):
        n = _apply(sg(i), p)
        n = tuple(v - __import__('math').floor(v) for v in n)
        dup = False
        for s in seen:
            d = [n[k] - s[k] for k in range(3)]
            d = [v - round(v) for v in d]
            if abs(d[0]) < 1e-6 and abs(d[1]) < 1e-6 and abs(d[2]) < 1e-6:
                dup = True
                break
        if not dup:
            seen.append(n)
            ops.append(i)
    return ops, (len(ops) == int(multiplicity))


def get_wyckoff_table(sg_info):
    """Wyckoff positions with everything a structure builder needs.

    Beyond the v7 fields (letter, multiplicity, site_symmetry, special_op) this
    emits the special operator as an exact rational projection matrix, its
    rank, and the coset operators. See the module header for why each is
    needed. Every field is guarded individually: a build whose Wyckoff objects
    lack one accessor still yields the rest.
    """
    positions = []
    table = _get_wyckoff_table_object(sg_info)   # unchanged helper from v7
    if table is None:
        return positions

    sg = sg_info.group()

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    for i in range(table.size()):
        try:
            pos = table.position(i)
        except Exception:
            continue

        site_sym = _safe(lambda: str(pos.point_group_type()), None)
        if site_sym is None:
            site_sym = _safe(lambda: str(pos.site_symmetry_group().point_group_type()), "?")

        mult = _safe(lambda: int(pos.multiplicity()), 0)
        entry = OrderedDict([
            ("letter", _safe(lambda: pos.letter(), "?")),
            ("multiplicity", mult),
            ("site_symmetry", site_sym),
            ("special_op", _safe(lambda: str(pos.special_op()), "")),
        ])

        sop = _safe(lambda: pos.special_op(), None)
        if sop is not None:
            try:
                r, t = sop.r(), sop.t()
                p_num = [int(v) for v in r.num()]
                p_den = int(r.den())
                t_num = [int(v) for v in t.num()]
                t_den = int(t.den())
                entry["P_num"] = p_num
                entry["P_den"] = p_den
                entry["T_num"] = t_num
                entry["T_den"] = t_den
                entry["n_free"] = _rank3(p_num, p_den)
            except Exception:
                entry["n_free"] = None
            try:
                ops, exact = _coset_ops(sg, sop, mult)
                entry["coset_ops"] = ops
                if not exact:
                    entry["coset_exact"] = False
            except Exception:
                pass

        positions.append(entry)

    return positions
