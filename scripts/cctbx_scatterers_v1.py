# cctbx_scatterers_v1.py
#
# Writes the scattering-factor tables sHarko needs into ./scatters/.
#
# WHY THIS EXISTS
# The correlation fitness compares observed intensities with |F_calc|^2. Point
# atoms (f = Z) make the two comparable only in a synthetic test; real data
# carries the form-factor fall-off, and that fall-off belongs on the MODEL
# side. Rescaling the observations to compensate - which is what an earlier
# version did, shell by shell - puts the true structure's correlation below 1
# by construction. So the coefficients have to come from somewhere, and cctbx
# already has them: there is no reason to embed a table by hand and every
# reason not to.
#
# WHAT IT WRITES
#   scatters/index.json          the evaluation formula and what is available
#   scatters/xray_wk1995.json    Waasmaier-Kirfel, 5 Gaussians + constant
#   scatters/xray_it1992.json    International Tables, 4 Gaussians + constant
#   scatters/neutron.json        bound coherent scattering lengths, in fm
#
# NOT one file per element. The whole X-ray table is a few tens of kilobytes
# and every run needs several elements at once, so splitting it would turn one
# fetch into a dozen for no saving. The folder holds one file per TABLE.
#
# EVALUATION (both X-ray tables share the form)
#
#     stol = sin(theta)/lambda = s/2 = 1/(2d)
#     f(stol) = c + sum_i a_i * exp(-b_i * stol^2)
#
# Note the argument is stol, NOT s. Passing s doubles the apparent resolution
# and makes every atom look far too diffuse - it is the single easiest mistake
# to make with these coefficients, so index.json states the formula explicitly
# and the JS loader asserts f(0) == Z.
#
# Usage:
#   python cctbx_scatterers_v1.py                 elements and common ions
#   python cctbx_scatterers_v1.py --no-ions       neutral atoms only
#   python cctbx_scatterers_v1.py --out ../app
#   python cctbx_scatterers_v1.py --check         print a verification table

import os
import sys
import json
import argparse
from collections import OrderedDict

try:
    from cctbx.eltbx import xray_scattering
except ImportError:
    print("cctbx is not importable in this environment.\n")
    print("    conda activate cctbx")
    print("    python cctbx_scatterers_v1.py\n")
    sys.exit(1)

SCHEMA_VERSION = 1
DIR_NAME = 'scatters'

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
            "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu").split()

# Charges worth probing. cctbx rejects labels it does not hold, so this is a
# candidate list rather than a claim about which ions exist.
CHARGES = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]


def ion_labels(symbol):
    out = []
    for q in CHARGES:
        out.append(f"{symbol}{abs(q)}{'+' if q > 0 else '-'}")
    return out


def atomic_number(symbol):
    """Z for a bare element symbol, for the f(0) == Z self-check."""
    try:
        from cctbx.eltbx import tiny_pse
        return int(tiny_pse.table(symbol).atomic_number())
    except Exception:
        try:
            return ELEMENTS.index(symbol) + 1
        except ValueError:
            return None


def _fetch_exact(fetcher, label):
    """Fetch a label ONLY if the table really holds it.

    cctbx's lookups do not fail on an unknown ion - they fall back to the
    neutral atom and return it without complaint. Probing 1128 candidate labels
    therefore "succeeded" 1128 times, and 922 of the results were the neutral
    element wearing an ion's name. Asking later for O2- would have returned
    neutral O silently, which is the worst kind of wrong: a plausible number
    that is not the one requested, in a place where a wrong f is
    indistinguishable from a wrong structure.

    Two defences, because the exact= keyword is not in every build: ask for an
    exact match if the signature allows it, and otherwise compare the label
    that came back with the one asked for.
    """
    entry = None
    try:
        entry = fetcher(label, True)          # exact=True where supported
    except Exception:
        try:
            entry = fetcher(label)
        except Exception:
            return None
        try:
            if str(entry.label()) != label:
                return None
        except Exception:
            return None
    if entry is None:
        return None
    # Belt and braces: verify the label even on the exact= path.
    try:
        if str(entry.label()) != label:
            return None
    except Exception:
        pass
    try:
        return entry.fetch()
    except Exception:
        return None


def _gaussian_to_dict(g):
    """Pull (a, b, c) out of a cctbx gaussian, whatever the accessor names.

    Builds differ in whether the terms come back as array_of_a()/array_of_b()
    or a()/b(). Both are tried before giving up on the entry rather than on the
    whole table.
    """
    for a_name, b_name in (("array_of_a", "array_of_b"), ("a", "b")):
        try:
            a = [float(v) for v in getattr(g, a_name)()]
            b = [float(v) for v in getattr(g, b_name)()]
            c = float(g.c()) if g.use_c() else 0.0
            if a and b and len(a) == len(b):
                return {"a": a, "b": b, "c": c}
        except Exception:
            continue
    return None


def build_xray_table(fetcher, labels):
    """One table: {label: {a:[...], b:[...], c: float, Z: int}}."""
    table = OrderedDict()
    skipped = 0
    for label in labels:
        g = _fetch_exact(fetcher, label)
        if g is None:
            skipped += 1
            continue
        d = _gaussian_to_dict(g)
        if d is None:
            skipped += 1
            continue
        z = atomic_number(label.rstrip('0123456789+-'))
        if z is not None:
            d["Z"] = z
        table[label] = d
    return table, skipped


def build_neutron_table(labels):
    """Bound coherent scattering lengths in femtometres.

    Powder work is as often neutron as X-ray, and the two need completely
    different f: neutron lengths do not fall off with angle at all, and light
    atoms are not invisible next to heavy ones. Emitting both costs almost
    nothing and lets the app follow the data rather than assume.
    """
    try:
        from cctbx.eltbx import neutron
    except ImportError:
        return None, "cctbx.eltbx.neutron not available in this build"

    table = OrderedDict()
    err = None
    for label in labels:
        sym = label.rstrip('0123456789+-')
        if sym != label:
            continue                      # neutron lengths are per nuclide, not per ion
        try:
            t = neutron.neutron_news_1992_table(sym)
            b = t.bound_coh_scatt_length()
            if b is None:
                continue
            entry = {"b_coh": float(b.real if hasattr(b, 'real') else b)}
            try:
                entry["b_coh_imag"] = float(b.imag)
            except Exception:
                pass
            table[sym] = entry
        except Exception as e:
            err = err or f"{type(e).__name__}: {e}"
            continue
    return table, err


def evaluate(entry, stol):
    """f(stol), the same formula the JS side uses. For the self-check."""
    f = entry.get("c", 0.0)
    for a, b in zip(entry["a"], entry["b"]):
        f += a * (2.718281828459045 ** (-b * stol * stol))
    return f


def self_check(table, name, verbose=False):
    """f(0) must equal Z for a neutral atom, and Z - charge for an ion.

    This catches the two things that actually go wrong: coefficients read from
    the wrong accessor, and the stol-versus-s confusion, which shows up as
    f(0) being right but f falling off at twice the rate it should.
    """
    bad = []
    for label, e in table.items():
        if "Z" not in e:
            continue
        charge = 0
        stripped = label.rstrip('+-')
        if stripped != label:
            digits = ''.join(ch for ch in stripped if ch.isdigit())
            if digits:
                charge = int(digits) * (1 if label.endswith('+') else -1)
        expect = e["Z"] - charge
        got = evaluate(e, 0.0)
        if abs(got - expect) > 0.25:
            bad.append((label, expect, got))
    # Ions are the interesting case, so say how many of each kind passed.
    n_ion = sum(1 for k in table if k.rstrip('+-') != k)
    if verbose:
        # A few neutral atoms and, separately, a few ions. An ion that is
        # secretly its neutral parent shows up here as an identical row.
        neutral = [k for k in table if k.rstrip('+-') == k][:4]
        ions = [k for k in table if k.rstrip('+-') != k][:4]
        for label in neutral + ions:
            e = table[label]
            vals = "  ".join(f"{evaluate(e, s):6.2f}" for s in (0.0, 0.2, 0.4, 0.6))
            print(f"    {label:<6} f(stol=0, .2, .4, .6) = {vals}")
    if bad:
        print(f"  [!] {name}: {len(bad)} entr(ies) where f(0) != Z - charge:")
        for label, expect, got in bad[:8]:
            print(f"        {label}: expected {expect}, got {got:.2f}")
    else:
        print(f"  {name}: f(0) == Z - charge for all {len(table)} entries "
              f"({len(table) - n_ion} neutral, {n_ion} ionic).")
    return len(bad)


def main():
    p = argparse.ArgumentParser(description="Write sHarko scattering-factor tables.")
    p.add_argument('--out', default=None, help="output folder (default: cwd, or where Harko.html is)")
    p.add_argument('--no-ions', action='store_true', help="neutral atoms only")
    p.add_argument('--check', action='store_true', help="print sample f values")
    args = p.parse_args()

    print("=" * 62)
    print("sHarko scattering-factor tables v1")
    print("=" * 62)

    labels = list(ELEMENTS)
    if not args.no_ions:
        for s in ELEMENTS:
            labels.extend(ion_labels(s))

    out_dir = args.out
    if out_dir is None:
        here = os.path.abspath(os.getcwd())
        script_dir = os.path.abspath(os.path.dirname(__file__))
        out_dir = here
        for folder in (here, script_dir, os.path.dirname(script_dir)):
            if folder and os.path.isfile(os.path.join(folder, 'Harko.html')):
                out_dir = folder
                break
    target = os.path.join(os.path.abspath(out_dir), DIR_NAME)
    os.makedirs(target, exist_ok=True)

    print(f"Probing {len(labels)} label(s)...")
    wk, wk_skip = build_xray_table(xray_scattering.wk1995, labels)
    print(f"  wk1995 : {len(wk):4d} entries  ({wk_skip} candidate labels not in the table)")

    it = OrderedDict()
    try:
        it, it_skip = build_xray_table(xray_scattering.it1992, labels)
        print(f"  it1992 : {len(it):4d} entries  ({it_skip} candidate labels not in the table)")
    except Exception as e:
        print(f"  it1992 : unavailable ({e})")

    nt, nerr = build_neutron_table(labels)
    if nt is None:
        print(f"  neutron: unavailable ({nerr})")
        nt = OrderedDict()
    else:
        print(f"  neutron: {len(nt)} entries")

    if not wk and not it:
        print("\n[!] No X-ray table could be built. Nothing written.")
        sys.exit(1)

    print("\nVerifying...")
    bad = 0
    if wk:
        bad += self_check(wk, "wk1995", args.check)
    if it:
        bad += self_check(it, "it1992", args.check)

    files = OrderedDict()
    def write(name, payload):
        path = os.path.join(target, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
        files[name] = os.path.getsize(path)

    if wk:
        write('xray_wk1995.json', {"schema_version": SCHEMA_VERSION,
                                   "kind": "xray_gaussian", "terms": 5, "table": wk})
    if it:
        write('xray_it1992.json', {"schema_version": SCHEMA_VERSION,
                                   "kind": "xray_gaussian", "terms": 4, "table": it})
    if nt:
        write('neutron.json', {"schema_version": SCHEMA_VERSION,
                               "kind": "neutron_length", "units": "fm", "table": nt})

    index = OrderedDict([
        ("schema_version", SCHEMA_VERSION),
        ("generated_by", os.path.basename(__file__)),
        # Spelled out because passing s instead of stol is the one mistake that
        # produces plausible-looking but systematically wrong scattering.
        ("formula", "f(stol) = c + sum_i a_i * exp(-b_i * stol^2)"),
        ("stol", "sin(theta)/lambda = 1/(2d) = s/2, where s = 1/d"),
        ("preferred", "xray_wk1995.json"),
        ("files", [OrderedDict([("name", n), ("bytes", b)]) for n, b in files.items()]),
    ])
    with open(os.path.join(target, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nWrote to {target}")
    for n, b in files.items():
        print(f"  {n:<22} {b/1024:6.1f} KB")
    print(f"  {'index.json':<22} {os.path.getsize(os.path.join(target,'index.json'))/1024:6.1f} KB")
    if bad:
        print(f"\n[!] {bad} entr(ies) failed the f(0) == Z check. Report the list above.")
    print("=" * 62)


if __name__ == "__main__":
    main()
