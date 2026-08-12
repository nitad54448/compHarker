# 1 Aug 2026 - adds the data needed to build a STRUCTURE, not just index peaks.
#
# NOTE ON LINEAGE: this file is v5 plus the additions listed below. It does NOT
# contain whatever changes went into your own v6 -- if that version altered the
# absence analysis, the Harker geometry or the output schema, merge those in
# here before running. The additions below are confined to three new helper
# functions and the block marked "NEW in v7" inside generate_all_space_groups();
# nothing else in v5 was touched, so a merge should be mechanical.
#
# Changes from v5:
#
# 1. NEW get_symmetry_operations(). The old get_rotations() kept only the
#    rotation part of each operator AND deduplicated on it, which loses
#    everything a structure solution needs:
#       - the translation parts (screw axes, glide planes) were thrown away,
#         so Pnma and Pmmm produced an identical list;
#       - deduplication on the rotation collapsed the centring operators, so
#         an I- or F-centred group lost half or three quarters of its
#         operators outright.
#    The new function emits all sg.order_z() operators with their translations,
#    as EXACT rationals (numerator/denominator) so no 1/3 ever arrives in the
#    browser as 0.333333.
#    "rotations" is still emitted, unchanged, so the existing JS keeps working.
#
# 2. NEW centring translations (n_ltr, ltr) and the conventional centring
#    letter, so a JS consumer can expand a primitive operator list.
#
# 3. NEW Wyckoff table per setting: letter, multiplicity, site symmetry and
#    the special-position operator. This is what turns "a peak at 0.12, 0.25,
#    0.31" into "Wyckoff 4c, site symmetry .m., x,1/4,z".
#
# 4. NEW laue_class and centering at the top level of each space group. The JS
#    (powder5.html) already reads currentSG.laue_class and currentSG.centering,
#    but v5 never wrote them -- sg_engine.js was deriving them.
#
# 5. NEW z_prime / order_z and is_chiral / is_centric flags.
#
# Everything from v5 is preserved byte for byte unless listed above.
#
# --- v5 header retained below ---
# 17 Jul 2026 - Rule-string ordering fix for the JS (brutus) parser
# 1. Diagonal glide rules now always emit the DOUBLED term first
#    ("2*l+h=4n" instead of "h+2*l=4n").
# 2. Removed the redundant axial rule alongside a diagonal 4n rule.
# - Fixed "Space group symbol not recognized" via the "Hall:" prefix.
# - Strict logic for d-glides (4n) and hhl zones.

# cctbx_Harko_v1.py
# Space-group database generator for sHarko.
#
# Lineage: this is cctbx_generate_sg_harker_v6.py with two stages replaced.
# Everything else - the systematic-absence analysis, the Harker geometry, the
# operator and centring extraction - is unchanged.
#
# CHANGE 1: the Wyckoff table is now machine-usable.
#
#   v6 emitted special_op as a STRING ("x,1/4,z"). Readable, but a browser
#   cannot project a trial coordinate onto a string, cannot count its degrees
#   of freedom, and cannot know which operators give DISTINCT images of a site
#   on it. All three are needed to search Wyckoff space instead of coordinate
#   space, which is what lets the swarm produce PbSO4 rather than Pb8S8O24.
#
#   Added per position:
#     P_num, P_den   rotation part of special_op as exact rationals. For any q,
#                    P.q + T lies ON the position. P is idempotent, so
#                    projecting twice is projecting once.
#     T_num, T_den   translation part, likewise exact.
#     n_free         rank(P): the true count of continuous parameters. 0 for
#                    4a, 2 for 4c (x,1/4,z), 3 for a general position.
#     coset_ops      indices into this setting's sym_ops giving exactly
#                    `multiplicity` distinct images. Because a Wyckoff position
#                    has one site-symmetry group throughout, these depend on
#                    the POSITION and not on where along it the atom sits - so
#                    they are precomputable, and the consumer needs no distance
#                    tolerance to decide how many atoms a site generates.
#
# CHANGE 2: one JSON file per space group, in ./sg/, plus a light index.
#
#   The single monolithic file held all 230 groups in all settings. Only one
#   setting is ever live, so the browser parsed the whole thing to use a
#   fraction of a percent of it. sg/index.json now carries just enough to build
#   a space-group picker (symbols, settings, Laue class), and sg/62.json is
#   fetched only when the user actually selects Pnma.
#
#   The monolithic file is STILL WRITTEN by default, because sg_engine.js has
#   not been migrated yet. Pass --no-legacy once it has.
#
# Usage:
#   python cctbx_Harko_v1.py                 all groups, ./sg/ plus legacy file
#   python cctbx_Harko_v1.py --only 62,206   just those two, for testing
#   python cctbx_Harko_v1.py --out ../app    write somewhere else
#   python cctbx_Harko_v1.py --no-legacy     skip the monolithic file
#   python cctbx_Harko_v1.py --pretty        indent the per-group files
#
# --- v6/v5 header retained below ---

import json
import os
from collections import OrderedDict, defaultdict
from cctbx import sgtbx
from fractions import Fraction
import sys
import argparse
import math


def get_rotations(sg_info):
    """Extracts all real-space rotational matrices for the space group.

    UNCHANGED from v5, and deliberately so: powder5.html may still read it.
    Do NOT use this for building a structure -- see get_symmetry_operations().
    """
    sg = sg_info.group()
    rots = []
    for i in range(sg.order_z()):
        r = tuple(sg(i).r().as_double())
        if r not in rots:
            rots.append(r)
    return rots


def _rational_triplet(num, den):
    """(numerators, denominator) -> list of 'p/q' strings in lowest terms."""
    out = []
    for n in num:
        f = Fraction(int(n), int(den))
        out.append(f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator))
    return out


def get_symmetry_operations(sg_info):
    """Full operator list: rotation, translation, and the xyz triplet.

    Returned per operator:
        xyz    : 'x,y,z' style string, exactly as cctbx formats it
        r      : 9 integers, row-major, x' = r . x + t
        t_num  : 3 integers, numerators of the translation
        t_den  : single integer denominator (cctbx uses 12, or 24 for some
                 d-glides); t = t_num / t_den exactly
        t      : the same translation as floats, for convenience

    Every operator of the group is listed, including the centring ones, so a
    consumer can apply the list directly with no further expansion.
    """
    sg = sg_info.group()
    ops = []
    for i in range(sg.order_z()):
        op = sg(i)
        r = op.r()
        t = op.t()
        r_num = [int(v) for v in r.num()]
        r_den = int(r.den())
        if r_den != 1:
            # cctbx always uses r_den == 1 for space-group rotations; guard
            # anyway so a silent non-integer matrix can never slip through.
            r_num = [v // r_den for v in r_num]
        t_num = [int(v) for v in t.num()]
        t_den = int(t.den())
        ops.append(OrderedDict([
            ("xyz", str(op)),
            ("r", r_num),
            ("t_num", t_num),
            ("t_den", t_den),
            ("t", [v / t_den for v in t_num]),
            ("t_frac", _rational_triplet(t_num, t_den)),
        ]))
    return ops


def get_centring(sg_info):
    """Centring translations as exact rationals, plus the conventional letter.

    Derived from the OPERATOR LIST rather than from sg.ltr(): the centring
    translations are exactly the translation parts of the operators whose
    rotation is the identity. The indexed accessor sg.ltr(i) is not available
    in every cctbx build ("_.ltr() takes 1 positional argument but 2 were
    given"), and there is no need for it -- this derivation is exact and works
    everywhere.
    """
    sg = sg_info.group()
    ltr = []
    seen = set()
    identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    for i in range(sg.order_z()):
        op = sg(i)
        r = op.r()
        r_num = tuple(int(v) for v in r.num())
        r_den = int(r.den())
        if r_den != 1:
            r_num = tuple(v // r_den for v in r_num)
        if r_num != identity:
            continue
        t = op.t()
        num = [int(x) for x in t.num()]
        den = int(t.den())
        key = tuple(Fraction(x, den) % 1 for x in num)
        if key in seen:
            continue
        seen.add(key)
        ltr.append(OrderedDict([
            ("t_num", num),
            ("t_den", den),
            ("t", [x / den for x in num]),
            ("t_frac", _rational_triplet(num, den)),
        ]))

    letter = "?"
    for getter in (
        lambda: sg.conventional_centring_type_symbol(),
        lambda: sg_info.type().lookup_symbol().strip().split(":")[0].strip()[0],
    ):
        try:
            value = getter()
            if value:
                letter = str(value).strip()[0]
                break
        except Exception:
            continue

    return letter, ltr


# cctbx does not expose the Wyckoff table at one fixed place. Depending on the
# build it is a method on space_group_info, or a submodule of sgtbx that
# "from cctbx import sgtbx" does NOT import for you (which is what produced
# "module 'cctbx.sgtbx' has no attribute 'wyckoff'" -- nothing to pip install,
# just an import that was never done). Try each in turn and remember which one
# worked so the probing happens once.
_WYCKOFF_ACCESSOR = "unprobed"


def _get_wyckoff_table_object(sg_info):
    global _WYCKOFF_ACCESSOR

    def _via_info(info):
        return info.wyckoff_table()

    def _via_submodule(info):
        from cctbx.sgtbx import wyckoff as _w
        return _w.table(info)

    def _via_import(info):
        import importlib
        _w = importlib.import_module("cctbx.sgtbx.wyckoff")
        return _w.table(info)

    def _via_attr(info):
        return sgtbx.wyckoff.table(info)

    def _via_flat(info):
        return sgtbx.wyckoff_table(info)

    routes = [
        ("space_group_info.wyckoff_table()", _via_info),
        ("from cctbx.sgtbx import wyckoff", _via_submodule),
        ("importlib cctbx.sgtbx.wyckoff", _via_import),
        ("sgtbx.wyckoff.table()", _via_attr),
        ("sgtbx.wyckoff_table()", _via_flat),
    ]

    # Once a route is known to work, stop probing.
    if _WYCKOFF_ACCESSOR not in ("unprobed", "none"):
        for name, fn in routes:
            if name == _WYCKOFF_ACCESSOR:
                try:
                    return fn(sg_info)
                except Exception:
                    break
    if _WYCKOFF_ACCESSOR == "none":
        return None

    errors = []
    for name, fn in routes:
        try:
            table = fn(sg_info)
            if table is not None:
                if _WYCKOFF_ACCESSOR == "unprobed":
                    print(f"\n[i] Wyckoff tables found via {name}.")
                    sys.stdout.flush()
                _WYCKOFF_ACCESSOR = name
                return table
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    _WYCKOFF_ACCESSOR = "none"
    print("\n[!] No Wyckoff accessor worked in this cctbx build. This is OPTIONAL:")
    print("    symops, centring and everything else are still written, and the")
    print("    structure builder derives site multiplicity from the operators")
    print("    itself. Only the ITA letters (4c, 8d, ...) will be missing.")
    for e in errors:
        print(f"      tried {e}")
    sys.stdout.flush()
    return None


def _rank3(num, den):
    """Exact rank of a 3x3 matrix given as 9 numerators over one denominator.

    Gaussian elimination over Fraction, not float: the projector for a position
    like (x,x,z) carries halves and thirds, and a rank misjudged by rounding
    would silently give a site the wrong number of free parameters.
    """
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


# A point with no rational relationship between its coordinates and no small
# denominators. Projecting it through special_op lands on a GENERIC point of the
# Wyckoff position; a tidier probe like (0.1, 0.2, 0.3) can land on a
# higher-symmetry sub-position and return too few coset operators.
_PROBE = (0.1357913, 0.2468135, 0.3791357)


def _apply_op(op, p):
    r = op.r().as_double()
    t = op.t().as_double()
    return (r[0]*p[0] + r[1]*p[1] + r[2]*p[2] + t[0],
            r[3]*p[0] + r[4]*p[1] + r[5]*p[2] + t[1],
            r[6]*p[0] + r[7]*p[1] + r[8]*p[2] + t[2])


def _coset_ops(sg, special_op, multiplicity):
    """Operator indices producing distinct images of a site on this position.

    Returns (ops, exact). `exact` is False when the count disagrees with the
    tabulated multiplicity, which means the probe point was unlucky; the field
    is still written but flagged, and a consumer should fall back to a
    distance-based coincidence test for that position alone.
    """
    p = _apply_op(special_op, _PROBE)
    seen, ops = [], []
    for i in range(sg.order_z()):
        n = _apply_op(sg(i), p)
        n = tuple(v - math.floor(v) for v in n)
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

    Beyond the v6 fields (letter, multiplicity, site_symmetry, special_op) this
    emits the special operator as an exact rational projection matrix, its rank,
    and the coset operators - see the module header for why each is needed.

    Every added field is guarded on its own. A cctbx build whose Wyckoff objects
    lack one accessor loses that field rather than the position, and a build
    with no Wyckoff table at all still produces a complete database minus the
    ITA letters, exactly as v6 did.
    """
    positions = []
    table = _get_wyckoff_table_object(sg_info)
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

        # point_group_type() lives directly on the position in some builds and
        # behind site_symmetry_group() in others; try both before giving up.
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
            except Exception as e:
                _warn_once("wyckoff special_op matrix", e)
            try:
                ops, exact = _coset_ops(sg, sop, mult)
                entry["coset_ops"] = ops
                if not exact:
                    entry["coset_exact"] = False
            except Exception as e:
                _warn_once("wyckoff coset operators", e)

        positions.append(entry)

    return positions


# ---------------------------------------------------------------------------
#  Everything below this line is unchanged from v5 except where marked.
# ---------------------------------------------------------------------------

def get_harker_geometry(sg_info):
    sg = sg_info.group()
    sections = []
    coords = ['u', 'v', 'w']
    vars_str = ['x', 'y', 'z']

    # Safely iterate through all symmetry operations
    for i in range(1, sg.order_z()):
        op = sg(i)
        r = op.r().as_double()
        t = op.t().as_double()

        # Construct R - I matrix
        r_minus_i = [
            [r[0]-1, r[1],   r[2]  ],
            [r[3],   r[4]-1, r[5]  ],
            [r[6],   r[7],   r[8]-1]
        ]

        # Identify zero rows (which define the Harker planes/lines)
        zero_rows = [idx for idx in range(3) if all(abs(val) < 1e-5 for val in r_minus_i[idx])]
        if not zero_rows:
            continue

        # Identify the submatrix to invert
        nz_rows = [idx for idx in range(3) if idx not in zero_rows]
        nz_cols = [idx for idx in range(3) if any(abs(r_minus_i[r_idx][idx]) > 1e-5 for r_idx in range(3))]

        solver = {"x": "?", "y": "?", "z": "?"}

        # Algebraically invert the submatrix to solve for x, y, z
        if len(nz_rows) == len(nz_cols) and len(nz_rows) > 0:
            n = len(nz_rows)
            A = [[r_minus_i[row][col] for col in nz_cols] for row in nz_rows]
            invA = None

            if n == 1:
                if abs(A[0][0]) > 1e-5:
                    invA = [[1.0 / A[0][0]]]
            elif n == 2:
                det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
                if abs(det) > 1e-5:
                    invA = [[A[1][1]/det, -A[0][1]/det], [-A[1][0]/det, A[0][0]/det]]
            elif n == 3:
                det = (A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                     - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                     + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]))
                if abs(det) > 1e-5:
                    invA = [
                        [(A[1][1]*A[2][2] - A[1][2]*A[2][1])/det, (A[0][2]*A[2][1] - A[0][1]*A[2][2])/det, (A[0][1]*A[1][2] - A[0][2]*A[1][1])/det],
                        [(A[1][2]*A[2][0] - A[1][0]*A[2][2])/det, (A[0][0]*A[2][2] - A[0][2]*A[2][0])/det, (A[0][2]*A[1][0] - A[0][0]*A[1][2])/det],
                        [(A[1][0]*A[2][1] - A[1][1]*A[2][0])/det, (A[0][1]*A[2][0] - A[0][0]*A[2][1])/det, (A[0][0]*A[1][1] - A[0][1]*A[1][0])/det]
                    ]

            # Format the output strings for the JS evaluator
            if invA:
                for c_idx, col_name in enumerate(nz_cols):
                    terms = []
                    for r_idx, row_name in enumerate(nz_rows):
                        coeff = invA[c_idx][r_idx]
                        if abs(coeff) > 1e-5:
                            terms.append(f"{coeff:g}*({coords[row_name]}-{t[row_name]:g})")
                    if terms:
                        solver[vars_str[col_name]] = " + ".join(terms).replace("+ -", "- ")

        sec_type = "plane" if len(zero_rows) == 1 else "line"

        for z_idx in zero_rows:
            val = t[z_idx] % 1.0
            if not any(s['coordinate'] == coords[z_idx] and abs(s['value'] - val) < 1e-5 for s in sections):
                sections.append({
                    "type": sec_type,
                    "coordinate": coords[z_idx],
                    "value": val,
                    "solver": solver
                })

    return sections


def evaluate_rule(h, k, l, rule_str):
    """Evaluates if a reflection satisfies a textual rule safely."""
    try:
        if "=" not in rule_str:
            return False

        lhs, rhs = rule_str.split("=")

        # Map the right-hand side to its modulus integer
        mod_map = {"2n": 2, "3n": 3, "4n": 4, "6n": 6}
        if rhs not in mod_map:
            return False

        mod_val = mod_map[rhs]

        # Evaluate the mathematical expression on the left-hand side natively
        context = {'h': h, 'k': k, 'l': l}
        lhs_val = eval(lhs, {}, context)

        return (lhs_val % mod_val) == 0
    except Exception:
        return False


def get_condition_string(present_values, axis_name):
    """Deduces axial conditions (2n, 4n, 6n)."""
    if not present_values:
        return None

    vals = sorted(list(set(present_values)))
    if not vals:
        return None

    # Strongest conditions first
    if all(v % 6 == 0 for v in vals): return f"{axis_name}=6n"
    if all(v % 4 == 0 for v in vals): return f"{axis_name}=4n"
    if all(v % 3 == 0 for v in vals): return f"{axis_name}=3n"
    if all(v % 2 == 0 for v in vals): return f"{axis_name}=2n"

    return None


def check_zonal_from_points(present_tuples, idx1, idx2):
    """
    Deduces zonal conditions, returning a list of all coexisting independent rules.
    """
    if not present_tuples:
        return []

    names = {0: 'h', 1: 'k', 2: 'l'}
    n1, n2 = names[idx1], names[idx2]

    v1 = [p[idx1] for p in present_tuples]
    v2 = [p[idx2] for p in present_tuples]
    sums = [p[idx1] + p[idx2] for p in present_tuples]
    diffs = [p[idx1] - p[idx2] for p in present_tuples]

    rules = []

    # --- 1. PRIORITY: Diamond/Complex Glides (4n) ---
    diag_forced_axis = None
    if all((2*p[idx1] + p[idx2]) % 4 == 0 for p in present_tuples):
        rules.append(f"2*{n1}+{n2}=4n")
        diag_forced_axis = n2
    elif all((2*p[idx1] - p[idx2]) % 4 == 0 for p in present_tuples):
        rules.append(f"2*{n1}-{n2}=4n")
        diag_forced_axis = n2
    elif all((p[idx1] + 2*p[idx2]) % 4 == 0 for p in present_tuples):
        rules.append(f"2*{n2}+{n1}=4n")   # doubled term first
        diag_forced_axis = n1

    # --- 2. PRIORITY: Standard Glide Sums (4n) ---
    has_4n_sum = False
    if all(s % 4 == 0 for s in sums):
        rules.append(f"{n1}+{n2}=4n")
        has_4n_sum = True
    elif all(d % 4 == 0 for d in diffs):
        rules.append(f"{n1}-{n2}=4n")

    # --- 3. PRIORITY: Axial Conditions (2n) ---
    has_n1_2n = all(v % 2 == 0 for v in v1)
    has_n2_2n = all(v % 2 == 0 for v in v2)

    if has_n1_2n and diag_forced_axis != n1:
        rules.append(f"{n1}=2n")
    if has_n2_2n and diag_forced_axis != n2:
        rules.append(f"{n2}=2n")

    # --- 4. PRIORITY: Weak Centering Sums (3n, 2n) ---
    if all(s % 3 == 0 for s in sums):
        rules.append(f"{n1}+{n2}=3n")
    elif all(d % 3 == 0 for d in diffs):
        rules.append(f"{n1}-{n2}=3n")

    if not (has_4n_sum or (has_n1_2n and has_n2_2n)):
        if all(s % 2 == 0 for s in sums):
            rules.append(f"{n1}+{n2}=2n")
        elif all(d % 2 == 0 for d in diffs):
            rules.append(f"{n1}-{n2}=2n")

    return rules


def is_redundant(specific_rule, general_rules, axes_indices):
    """Checks if a specific rule is already covered by general HKL rules."""
    if not general_rules:
        return False

    valid_points_under_general = []
    is_diagonal = axes_indices.get('diagonal', False)
    active_axes = [key for key, v in axes_indices.items() if v is None and key != 'diagonal']

    for i in range(1, 25):
        if len(active_axes) == 1 and not is_diagonal:  # Axial check
            h, k, l = 0, 0, 0
            if active_axes[0] == 'h': h = i
            elif active_axes[0] == 'k': k = i
            elif active_axes[0] == 'l': l = i

            if all(evaluate_rule(h, k, l, r) for r in general_rules):
                valid_points_under_general.append((h, k, l))

        elif len(active_axes) >= 2 or is_diagonal:  # Zonal check
            for j in range(1, 25):
                h, k, l = 0, 0, 0
                if is_diagonal:  # hhl
                    h, k, l = i, i, j
                elif axes_indices['l'] == 0:  # hk0
                    h, k = i, j
                elif axes_indices['k'] == 0:  # h0l
                    h, l = i, j
                elif axes_indices['h'] == 0:  # 0kl
                    k, l = i, j

                if all(evaluate_rule(h, k, l, r) for r in general_rules):
                    valid_points_under_general.append((h, k, l))

    if not valid_points_under_general:
        return False

    if len(active_axes) == 1 and not is_diagonal:
        idx_map = {'h': 0, 'k': 1, 'l': 2}
        axis_char = active_axes[0]
        vals = [p[idx_map[axis_char]] for p in valid_points_under_general]
        return get_condition_string(vals, axis_char) == specific_rule

    else:
        idx_map = {'h': 0, 'k': 1, 'l': 2}
        idx1 = 0 if is_diagonal else idx_map[active_axes[0]]
        idx2 = 2 if is_diagonal else idx_map[active_axes[1]]
        return specific_rule in check_zonal_from_points(valid_points_under_general, idx1, idx2)


def analyze_systematic_absences(space_group_info):
    sg = space_group_info.group()

    # --- 1. HKL Conditions (Centering) ---
    present_hkl = []
    for h in range(1, 8):
        for k in range(1, 8):
            for l in range(1, 8):
                if not sg.is_sys_absent((h, k, l)):
                    present_hkl.append((h, k, l))

    hkl_rules = []
    if present_hkl:
        if all((h+k) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+k=2n")
        if all((h+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+l=2n")
        if all((k+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("k+l=2n")
        if all((h+k+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+k+l=2n")

        # Rhombohedral check (Obverse vs Reverse)
        if all((-h+k+l) % 3 == 0 for h, k, l in present_hkl): hkl_rules.append("-h+k+l=3n")
        if all((h-k+l) % 3 == 0 for h, k, l in present_hkl): hkl_rules.append("h-k+l=3n")

        # F-centering cleanup (only if strictly F and not I)
        if "h+k=2n" in hkl_rules and "h+l=2n" in hkl_rules and "k+l=2n" in hkl_rules:
            if "h+k+l=2n" not in hkl_rules:
                hkl_rules = ["h+k=2n", "h+l=2n", "k+l=2n"]

    conditions = OrderedDict()
    if hkl_rules:
        conditions['hkl'] = hkl_rules

    # --- 2. Zonal Conditions ---
    zones = [
        ('0kl', {'h': 0, 'k': None, 'l': None}, 1, 2),
        ('h0l', {'h': None, 'k': 0, 'l': None}, 0, 2),
        ('hk0', {'h': None, 'k': None, 'l': 0}, 0, 1),
        ('hhl', {'h': None, 'k': None, 'l': None, 'diagonal': True}, 0, 2)
    ]

    for zone_name, axes_map, idx1, idx2 in zones:
        points = []
        for i in range(1, 20):
            for j in range(1, 20):
                hkl = [0, 0, 0]
                if zone_name == 'hhl':
                    hkl[0] = i
                    hkl[1] = i
                    hkl[2] = j
                else:
                    hkl[idx1] = i
                    hkl[idx2] = j

                if not sg.is_sys_absent(tuple(hkl)):
                    points.append(tuple(hkl))

        rules = check_zonal_from_points(points, idx1, idx2)
        if rules:
            valid_rules = [r for r in rules if not is_redundant(r, hkl_rules, axes_map)]
            if valid_rules:
                conditions[zone_name] = valid_rules

    # --- 3. Axial Conditions ---
    axes_defs = [
        ('h00', {'h': None, 'k': 0, 'l': 0}, 0, 'h'),
        ('0k0', {'h': 0, 'k': None, 'l': 0}, 1, 'k'),
        ('00l', {'h': 0, 'k': 0, 'l': None}, 2, 'l')
    ]

    for axis_name, axes_map, idx, label in axes_defs:
        points = []
        for i in range(1, 30):
            hkl = [0, 0, 0]
            hkl[idx] = i
            if not sg.is_sys_absent(tuple(hkl)):
                points.append(hkl[idx])

        rule = get_condition_string(points, label)
        if rule:
            is_covered = False
            if is_redundant(rule, hkl_rules, axes_map):
                is_covered = True

            if not is_covered:
                parents = []
                if axis_name == 'h00': parents = ['hk0', 'h0l']
                elif axis_name == '0k0': parents = ['hk0', '0kl']
                elif axis_name == '00l': parents = ['h0l', '0kl', 'hhl']

                for p in parents:
                    if p in conditions:
                        if is_redundant(rule, conditions[p], axes_map):
                            is_covered = True
                            break

            if not is_covered:
                conditions[axis_name] = [rule]

    return conditions


_WARNED = set()


def _warn_once(what, err):
    """Report an unavailable cctbx accessor once, not 700 times."""
    key = f"{what}:{type(err).__name__}:{err}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"\n[!] {what} unavailable in this cctbx build ({err}). "
          f"That field will be empty; everything else is still written.")
    sys.stdout.flush()


def generate_all_space_groups(only=None):
    all_data = defaultdict(lambda: {
        "number": 0, "standard_symbol": "", "crystal_system": "",
        "point_group": "", "laue_class": "", "centrosymmetric": False,
        "chiral": False, "settings": []
    })

    iterator = sgtbx.space_group_symbol_iterator()
    processed_settings = defaultdict(set)

    print("Processing settings...")
    count = 0

    while True:
        try:
            symbols = iterator.next()
        except StopIteration:
            break

        if symbols.number() == 0:
            break

        try:
            sg_num = symbols.number()
            # --only: skip everything else before any real work is done.
            if only is not None and sg_num not in only:
                continue
            hm_symbol = symbols.hermann_mauguin()
            hall = symbols.hall().strip()

            uid = hall
            if uid in processed_settings[sg_num]:
                continue
            processed_settings[sg_num].add(uid)

            sg_info = sgtbx.space_group_info(symbol=f"Hall: {hall}")
            sg = sg_info.group()

            if all_data[str(sg_num)]["number"] == 0:
                all_data[str(sg_num)]["number"] = sg_num
                all_data[str(sg_num)]["standard_symbol"] = sg_info.type().lookup_symbol()
                all_data[str(sg_num)]["crystal_system"] = str(sg.crystal_system()).lower()
                all_data[str(sg_num)]["point_group"] = str(sg.point_group_type())
                all_data[str(sg_num)]["centrosymmetric"] = sg.is_centric()
                # NEW in v7 -- powder5.html already reads currentSG.laue_class
                try:
                    all_data[str(sg_num)]["laue_class"] = str(sg.laue_group_type())
                except Exception as e:
                    _warn_once("laue_group_type", e)
                    all_data[str(sg_num)]["laue_class"] = ""
                try:
                    all_data[str(sg_num)]["chiral"] = bool(sg.is_chiral())
                except Exception:
                    all_data[str(sg_num)]["chiral"] = False

            conditions = analyze_systematic_absences(sg_info)
            harker_data = get_harker_geometry(sg_info)
            rotations = get_rotations(sg_info)

            # --- NEW in v7 ---
            # Each of these is isolated. The enclosing try/except skips the
            # WHOLE setting on any exception, so an API difference in one
            # accessor used to silently drop every space group in the file
            # (all 230 of them, leaving an empty JSON). A failure here now
            # costs only the field it belongs to.
            try:
                symops = get_symmetry_operations(sg_info)
            except Exception as e:
                _warn_once("sym_ops", e)
                symops = []
            try:
                centring_letter, centring_ltr = get_centring(sg_info)
            except Exception as e:
                _warn_once("centring", e)
                centring_letter, centring_ltr = "?", []
            try:
                wyckoff = get_wyckoff_table(sg_info)
            except Exception as e:
                _warn_once("wyckoff", e)
                wyckoff = []
            try:
                order_z = int(sg.order_z())
            except Exception:
                order_z = len(symops)
            try:
                order_p = int(sg.order_p())
            except Exception:
                order_p = 0

            desc = symbols.qualifier() if symbols.qualifier() else "standard"
            clean_sym = hm_symbol.replace(" ", "")

            all_data[str(sg_num)]["settings"].append({
                "symbol": clean_sym,
                "description": desc,
                "hall": hall,
                "reflection_conditions": conditions,
                "harker_sections": harker_data,
                "rotations": rotations,
                # --- NEW in v7 ---
                "sym_ops": symops,
                "order_z": order_z,
                "order_p": order_p,
                "centering": centring_letter,
                "centring_translations": centring_ltr,
                "wyckoff": wyckoff,
            })

            count += 1
            if count % 50 == 0:
                print(f"Processed {count} settings...", end="\r")
                sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error processing SG {sg_num}: {e}")
            sys.stdout.flush()
            continue

    return all_data


SCHEMA_VERSION = 8

# Legacy monolithic file. Still fetched by sg_engine.js and refinement_worker.js
# until those migrate to the sg/ folder, so it is written unless --no-legacy.
LEGACY_OUTPUT_NAME = 'cctbx_space_groups_all_settings_v6.json'

SG_DIR_NAME = 'sg'
INDEX_NAME = 'index.json'


def resolve_output_dir(explicit):
    """Where to write, stated in absolute terms.

    Running from a scripts/ subfolder used to drop the JSON there while the dev
    server served the project root, giving a 404 for a file that had just been
    written successfully. The app's own location decides, unless overridden.

    v6 looked only for powder5.html; Harko.html is checked first now, since that
    is what the app is called.
    """
    if explicit:
        target = os.path.abspath(explicit)
        os.makedirs(target, exist_ok=True)
        print(f"Output folder given on the command line: {target}")
        return target

    here = os.path.abspath(os.getcwd())
    script_dir = os.path.abspath(os.path.dirname(__file__))
    for folder in (here, script_dir, os.path.dirname(script_dir), os.path.dirname(here)):
        if not folder:
            continue
        for app in ('Harko.html', 'powder5.html'):
            if os.path.isfile(os.path.join(folder, app)):
                if os.path.abspath(folder) != here:
                    print(f"\n[i] {app} found in {folder}")
                    print("    Writing there, not into the current folder, because the app")
                    print("    fetches with a relative path from its own location.")
                return folder

    print("\n[!] Neither Harko.html nor powder5.html was found nearby, so the output")
    print(f"    goes in the current folder ({here}). If the browser reports a 404,")
    print("    move the sg/ folder next to Harko.html, or re-run as:")
    print(f"        python {os.path.basename(__file__)} --out <folder-with-Harko.html>")
    return here


def write_split_database(sorted_data, out_dir, pretty=False):
    """One file per space-group NUMBER in sg/, plus a light index.

    Per NUMBER rather than per SETTING: the app picks a number first and offers
    its settings afterwards, so a file holding every setting of one group is
    exactly one fetch per user choice. Splitting per setting would mean the app
    could not list the settings without fetching them all.

    The index carries enough to populate the picker - symbol, crystal system,
    Laue class and a one-line summary of each setting - so nothing is fetched
    until a group is actually selected.
    """
    sg_dir = os.path.join(out_dir, SG_DIR_NAME)
    os.makedirs(sg_dir, exist_ok=True)

    dump_kw = dict(ensure_ascii=False)
    if pretty:
        dump_kw['indent'] = 2
    else:
        # Compact by default: these files are machine-read, and dropping the
        # indentation takes roughly a third off the bytes the browser fetches.
        dump_kw['separators'] = (',', ':')

    index = OrderedDict()
    index["schema_version"] = SCHEMA_VERSION
    index["generated_by"] = os.path.basename(__file__)
    index["space_groups"] = OrderedDict()

    total_bytes = 0
    biggest = ("", 0)
    for num, entry in sorted_data.items():
        payload = OrderedDict()
        payload["schema_version"] = SCHEMA_VERSION
        for k, v in entry.items():
            payload[k] = v

        path = os.path.join(sg_dir, f"{num}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, **dump_kw)
        size = os.path.getsize(path)
        total_bytes += size
        if size > biggest[1]:
            biggest = (f"{num}.json", size)

        index["space_groups"][num] = OrderedDict([
            ("number", entry["number"]),
            ("standard_symbol", entry["standard_symbol"]),
            ("crystal_system", entry["crystal_system"]),
            ("point_group", entry["point_group"]),
            ("laue_class", entry.get("laue_class", "")),
            ("centrosymmetric", entry["centrosymmetric"]),
            ("chiral", entry.get("chiral", False)),
            ("file", f"{SG_DIR_NAME}/{num}.json"),
            ("settings", [OrderedDict([
                ("symbol", s["symbol"]),
                ("description", s["description"]),
                ("hall", s["hall"]),
                ("centering", s.get("centering", "?")),
                ("order_z", s.get("order_z", 0)),
                ("n_wyckoff", len(s.get("wyckoff", []))),
            ]) for s in entry["settings"]]),
        ])

    # One compact file with everything, alongside the split ones. The browser
    # needs the whole database in memory anyway (the picker lists all 230, and
    # the analysis paths are synchronous), so reassembling it from 230 requests
    # is work for no benefit - the saving over the legacy monolith was never the
    # splitting, it was dropping the indentation. The split files stay for any
    # consumer that wants one group at a time.
    all_payload = OrderedDict([("schema_version", SCHEMA_VERSION),
                               ("generated_by", os.path.basename(__file__)),
                               ("space_groups", sorted_data)])
    all_path = os.path.join(sg_dir, 'all.json')
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(all_payload, f, ensure_ascii=False, separators=(',', ':'))

    index_path = os.path.join(sg_dir, INDEX_NAME)
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(sorted_data)} group files to {sg_dir}")
    print(f"  all.json  {os.path.getsize(all_path)/1024:.0f} KB  (single-request fast path)")
    print(f"  total {total_bytes/1024:.0f} KB, largest {biggest[0]} at {biggest[1]/1024:.0f} KB")
    print(f"  index {os.path.getsize(index_path)/1024:.0f} KB ({INDEX_NAME})")
    return sg_dir


def verify(sorted_data):
    """Operator lists must be closed groups; coset counts must match."""
    print("\nVerifying operator lists form closed groups...")
    bad = checked = missing_symops = missing_wyckoff = 0
    coset_bad = []
    nfree_missing = 0

    for num, entry in sorted_data.items():
        for setting in entry["settings"]:
            ops = setting["sym_ops"]
            wy = setting.get("wyckoff") or []
            if not wy:
                missing_wyckoff += 1
            for w in wy:
                if w.get("coset_exact") is False or (
                        "coset_ops" in w and len(w["coset_ops"]) != w["multiplicity"]):
                    coset_bad.append(f"SG {num} {setting['symbol']} "
                                     f"{w['multiplicity']}{w['letter']}")
                if "n_free" not in w:
                    nfree_missing += 1
            if not ops:
                missing_symops += 1
                continue
            if len(ops) != setting["order_z"]:
                print(f"  [!] SG {num} {setting['symbol']}: {len(ops)} ops but order_z={setting['order_z']}")
                bad += 1
                continue
            keys = set()
            for op in ops:
                keys.add((tuple(op["r"]), tuple(Fraction(n, op["t_den"]) % 1 for n in op["t_num"])))
            if len(keys) != len(ops):
                print(f"  [!] SG {num} {setting['symbol']}: duplicate operators")
                bad += 1
                continue
            closed = True
            for a in ops:
                for b in ops:
                    ra, ta, da = a["r"], a["t_num"], a["t_den"]
                    rb, tb, db = b["r"], b["t_num"], b["t_den"]
                    rc = [sum(ra[3 * i + k] * rb[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]
                    tc = [Fraction(ta[i], da) + sum(Fraction(ra[3 * i + k] * tb[k], db) for k in range(3))
                          for i in range(3)]
                    if (tuple(rc), tuple(t % 1 for t in tc)) not in keys:
                        closed = False
                        break
                if not closed:
                    break
            if not closed:
                print(f"  [!] SG {num} {setting['symbol']}: operator list is not closed")
                bad += 1
            checked += 1

    print(f"Checked {checked} settings, {bad} problem(s).")
    if missing_symops:
        print(f"[!] {missing_symops} setting(s) have NO symops -- structure building "
              f"will not work for those. Scroll up for the accessor that failed.")
    if missing_wyckoff:
        print(f"[!] {missing_wyckoff} setting(s) have no Wyckoff table. The Wyckoff "
              f"search cannot run for those; the rest of the app still works.")
    if nfree_missing:
        print(f"[!] {nfree_missing} Wyckoff position(s) have no n_free -- the special_op "
              f"matrix accessor failed. Scroll up for the reason.")
    if coset_bad:
        print(f"[!] {len(coset_bad)} position(s) gave a coset count that disagrees with "
              f"the tabulated multiplicity:")
        for c in coset_bad[:10]:
            print(f"      {c}")
        if len(coset_bad) > 10:
            print(f"      ... and {len(coset_bad) - 10} more")
        print("    Those positions carry coset_exact:false and the consumer will fall")
        print("    back to a distance test for them. Report the list.")
    if not (missing_symops or coset_bad or nfree_missing):
        print("All settings carry a full closed operator list and an exact Wyckoff table.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate the sHarko space-group database.")
    p.add_argument('--out', default=None,
                   help="output folder (default: the folder containing Harko.html, else cwd)")
    p.add_argument('--only', default=None,
                   help="comma-separated space-group numbers, e.g. 62,206 - for quick tests")
    p.add_argument('--pretty', action='store_true',
                   help="indent the per-group files (about 50%% larger)")
    p.add_argument('--no-legacy', action='store_true',
                   help="skip the monolithic cctbx_space_groups_all_settings_v6.json")
    p.add_argument('--no-check', action='store_true',
                   help="skip the group-closure verification (it is the slow part)")
    return p.parse_args()


def main():
    args = parse_args()

    only = None
    if args.only:
        only = set(int(x) for x in args.only.replace(' ', '').split(',') if x)

    print("=" * 62)
    print("sHarko space-group generator v1 - Wyckoff projections, split output")
    print("=" * 62)
    if only:
        print(f"Restricted to space group(s): {sorted(only)}")

    data = generate_all_space_groups(only=only)

    total_settings = sum(len(x['settings']) for x in data.values())
    print(f"\nFinal processing complete.")
    print(f"Total space group numbers: {len(data)}")
    print(f"Total settings: {total_settings}")

    sorted_data = OrderedDict()
    for k in sorted(data.keys(), key=int):
        sorted_data[k] = data[k]
        sorted_data[k]["settings"].sort(key=lambda x: x["symbol"])

    out_dir = resolve_output_dir(args.out)
    write_split_database(sorted_data, out_dir, pretty=args.pretty)

    if not args.no_legacy:
        wrapper = OrderedDict()
        wrapper["schema_version"] = SCHEMA_VERSION
        wrapper["space_groups"] = sorted_data
        legacy = os.path.join(out_dir, LEGACY_OUTPUT_NAME)
        with open(legacy, 'w', encoding='utf-8') as f:
            json.dump(wrapper, f, indent=2, ensure_ascii=False)
        print(f"\nAlso wrote the legacy monolithic file:")
        print(f"  {legacy}  ({os.path.getsize(legacy)/1024/1024:.1f} MB)")
        print("  sg_engine.js and refinement_worker.js still fetch this. Once they")
        print("  read sg/index.json instead, re-run with --no-legacy.")

    if not args.no_check:
        verify(sorted_data)
    else:
        print("\n[i] Verification skipped (--no-check).")

    print("=" * 62)


if __name__ == "__main__":
    main()
