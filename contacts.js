// Bumped whenever this file changes. Harko.html compares it against what it
// expects and says so if a browser cache is serving something older.
const SHARKO_CONTACTS_VERSION = '2026-08-09j';

/* ------------------------------------------------------------------
   Interatomic distances for a candidate structure.

   Two uses, and they want different things.

   Ranking. When several assignments fit the intensities within a hundredth of
   a correlation unit - which is the normal state of affairs, not a failure -
   the map has stopped discriminating and chemistry has to. A model with a
   1.4 A Pb-O contact is wrong however well it correlates, and one whose S-O
   distances cluster at 1.48 A is right in a way the correlation cannot see.

   Inspection. Given a site, what is around it and how far. This is the
   question a crystallographer asks of any proposed structure, and answering it
   turns a coordinate table from a list of numbers into something judgeable.

   Both are computed from the same symmetry expansion, which is why they live
   together.
   ------------------------------------------------------------------ */

/* ------------------------------------------------------------------
   Tunables. Named because two of them are tolerances that look
   interchangeable and are not: one is float noise, the other is chemistry.
   ------------------------------------------------------------------ */
const CT_DEFAULTS = Object.freeze({
    // Fractional coordinate close enough to 1 to BE 1. This is float noise on a
    // symmetry image, not a distance - it exists so images of a special
    // position deduplicate through the Set key below. Fractional rather than
    // Cartesian is right here for exactly that reason: it is a rounding
    // question, not a chemical one.
    wrapEpsilonFrac: 0.999,
    dedupeDecimals: 3,
    // Fractional bounding box used to reject far-apart pairs before the
    // Cartesian distance is computed. Purely a speed filter; it must be loose
    // enough never to discard a genuine near neighbour, which at one shell of
    // neighbouring cells it is.
    pairBoxFrac: 1.2,
    contactCutoffA: 3.6,        // neighbour list radius
    maxNeighbours: 12,
    qualityCutoffA: 3.2,        // radius the bond-valence sum runs over
    // First coordination shell as a multiple of the SHORTEST contact. A flat
    // radius counts whatever lies beyond the real polyhedron: at 3.2 A a
    // sulfate sulfur came back nine-coordinate with a mean bond of 2.7 A.
    shellFactor: 1.35,
    // Brese & O'Keeffe universal bond-valence softness parameter.
    bondValenceB: 0.37
});

/**
 * Expands the asymmetric unit into the cell, plus one shell of neighbouring
 * cells.
 *
 * ONE SHELL IS ENOUGH, INCLUDING FOR SKEWED CELLS. Unlike the naive
 * minimum-image convention this does not round each fractional component
 * independently - it enumerates the neighbouring translations and takes the
 * true minimum - so it is correct where `d - round(d)` is not. Checked
 * numerically against an exhaustive +/-4 shell for monoclinic beta = 125,
 * triclinic 70/115/100, rhombohedral alpha = 60 and monoclinic beta = 150:
 * exact in every case, including with the bounding-box filter applied. This
 * routine is therefore the reference the GPU kernel's reduced-basis distance
 * is expected to agree with.
 *
 * The neighbouring cells matter: a site at z = 0.98 and one at z = 0.02 are
 * 0.04 apart through the cell face, not 0.96 the long way round. Working only
 * inside [0,1) and taking the minimum image works for a distance, but not for
 * "list what is near this atom" - the same neighbour would be reported at its
 * far-side coordinates and look absurd.
 *
 * @param sites  [{element, x, y, z}] the asymmetric unit
 * @param symOps [{r:[9], t:[3]}]
 * @param range  how many cells out; 1 is enough for any contact under ~5 A
 */
function expandForContacts(sites, symOps, range = 1) {
    const inCell = [];
    const seen = new Set();
    sites.forEach((s, si) => {
        for (const op of symOps) {
            let x = s.x * op.r[0] + s.y * op.r[1] + s.z * op.r[2] + op.t[0];
            let y = s.x * op.r[3] + s.y * op.r[4] + s.z * op.r[5] + op.t[1];
            let z = s.x * op.r[6] + s.y * op.r[7] + s.z * op.r[8] + op.t[2];
 

            x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
            
            // Snap floating point boundary values near 1.0 back to 0.0
            // so they deduplicate correctly in the Set.
            const eps = CT_DEFAULTS.wrapEpsilonFrac;
            if (x > eps) x = 0;
            if (y > eps) y = 0;
            if (z > eps) z = 0;

            // Images of a site on a special position coincide exactly; keeping
            // both would report a contact of 0 A with itself.
            const dp = CT_DEFAULTS.dedupeDecimals;
            const key = `${si}:${x.toFixed(dp)},${y.toFixed(dp)},${z.toFixed(dp)}`;
 
            if (seen.has(key)) continue;
            seen.add(key);
            inCell.push({ site: si, element: s.element, x, y, z });
        }
    });

    const out = [];
    for (let a = -range; a <= range; a++)
        for (let b = -range; b <= range; b++)
            for (let c = -range; c <= range; c++)
                for (const p of inCell)
                    out.push({ site: p.site, element: p.element,
                               x: p.x + a, y: p.y + b, z: p.z + c,
                               home: (a === 0 && b === 0 && c === 0) });
    return { inCell, all: out };
}

function cartDistance(dx, dy, dz, orth) {
    const X = orth[0] * dx + orth[1] * dy + orth[2] * dz;
    const Y = orth[3] * dx + orth[4] * dy + orth[5] * dz;
    const Z = orth[6] * dx + orth[7] * dy + orth[8] * dz;
    return Math.sqrt(X * X + Y * Y + Z * Z);
}

/**
 * Neighbours of one asymmetric-unit site, nearest first.
 *
 * Only images in the home cell are used as the reference atom - the others are
 * the same atom seen from a different cell, and listing each of them would
 * repeat every contact 27 times.
 */
function contactsForSite(siteIdx, sites, symOps, orth, options = {}) {
    const cutoff = options.cutoff ?? CT_DEFAULTS.contactCutoffA;
    const maxN = options.max ?? CT_DEFAULTS.maxNeighbours;
    const { inCell, all } = expandForContacts(sites, symOps, 1);
    const BOX = CT_DEFAULTS.pairBoxFrac;

    const ref = inCell.find(p => p.site === siteIdx);
    if (!ref) return [];

    const out = [];
    for (const q of all) {
        const dx = q.x - ref.x, dy = q.y - ref.y, dz = q.z - ref.z;
        if (Math.abs(dx) > BOX || Math.abs(dy) > BOX || Math.abs(dz) > BOX) continue;
        const d = cartDistance(dx, dy, dz, orth);
        if (d < 1e-4 || d > cutoff) continue;      // 0 is the atom itself
        out.push({ element: q.element, site: q.site, d,
                   symmetryRelated: q.site === siteIdx });
    }
    out.sort((a, b) => a.d - b.d);
    return out.slice(0, maxN);
}

/**
 * Whole-structure contact summary, for ranking candidates.
 *
 * `windows` are the user's distance restraints, in the same shape the search
 * uses. Reporting how each is met is more useful than a single pass/fail: a
 * structure missing one S-O contact by 0.05 A is a different proposition from
 * one whose sulfur has no oxygen within 3 A.
 */
function contactSummary(sites, symOps, orth, windows = [], options = {}) {
    const { inCell, all } = expandForContacts(sites, symOps, 1);
    const BOX = CT_DEFAULTS.pairBoxFrac;
    // Per-pair floor below which a contact is not close, it is impossible.
    // Distinct from the user's windows: those express what a chemist expects,
    // this expresses what atoms permit. A structure with lead and oxygen 0.32 A
    // apart is not a worse answer than one at 2.6 A, it is not an answer, and
    // it must be excluded whether or not anyone thought to type a window.
    const floorOf = options.floor || (() => 0);

    let shortest = Infinity, shortestPair = null;
    let impossible = 0, worstOverlap = 0, impossiblePair = null;
    for (const p of inCell) {
        for (const q of all) {
            const dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            if (Math.abs(dx) > BOX || Math.abs(dy) > BOX || Math.abs(dz) > BOX) continue;
            const d = cartDistance(dx, dy, dz, orth);
            if (d < 1e-4) continue;
            if (d < shortest) { shortest = d; shortestPair = `${p.element}-${q.element}`; }
            const fl = floorOf(p.element, q.element);
            if (fl > 0 && d < fl) {
                impossible++;
                const over = (fl - d) / fl;
                if (over > worstOverlap) {
                    worstOverlap = over;
                    impossiblePair = `${p.element}-${q.element} ${d.toFixed(2)} A (floor ${fl.toFixed(2)})`;
                }
            }
        }
    }
    impossible = Math.round(impossible / 2);   // each pair was met from both ends

    // Distance constraints, evaluated exactly as the kernel evaluates them, so
    // the table cannot say a structure is fine while the search was charging it.
    //
    //   dmin  a HARD FLOOR - no B may be closer than this to any A, whether or
    //         not a count was asked for.
    //   count how many B lie in [dmin, dmax] around every A. Exact by default,
    //         "at least" when countMode is 'atleast'.
    //   dmax  with no count, the nearest-neighbour condition: every A must have
    //         SOME B within it.
    const checks = [];
    for (const w of windows) {
        const A = String(w.a).toUpperCase(), B = String(w.b).toUpperCase();
        const refs = inCell.filter(p => p.element.toUpperCase() === A);
        const hasCount = Number.isFinite(w.count) && w.count >= 0;
        const dmin = Number.isFinite(w.dmin) ? w.dmin : null;
        const dmax = Number.isFinite(w.dmax) ? w.dmax : null;

        if (!refs.length) {
            checks.push({ a: A, b: B, dmin, dmax, count: hasCount ? w.count : null,
                          countMode: w.countMode || null, n: 0, violations: 0, worst: 0,
                          note: `no ${A} in the cell` });
            continue;
        }

        let violations = 0, worst = 0, floorBreaks = 0, closest = Infinity;
        const nearest = [], counts = [], nearestN = [];
        for (const p of refs) {
            let best = Infinity, inWindow = 0;
            // The N nearest partners, the same quantity the kernel charges on.
            // "0 of 4" says a constraint is unmet and nothing about how unmet:
            // four partners at 2.00 A against a window ending at 1.90 reads
            // identically to four at 4 A, and they are completely different
            // situations - one is a converged structure a tenth of an Angstrom
            // out, the other is noise.
            const ds = [];
            for (const q of all) {
                if (q.element.toUpperCase() !== B) continue;
                const dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
                if (Math.abs(dx) > BOX || Math.abs(dy) > BOX || Math.abs(dz) > BOX) continue;
                const d = cartDistance(dx, dy, dz, orth);
                if (d < 1e-4) continue;
                if (d < best) best = d;
                ds.push(d);
                if (dmin !== null && d < dmin - 1e-3) { floorBreaks++; closest = Math.min(closest, d); }
                if ((dmin === null || d >= dmin) && (dmax === null || d <= dmax)) inWindow++;
            }
            if (best === Infinity) continue;
            nearest.push(best);
            counts.push(inWindow);
            if (hasCount) {
                ds.sort((x, y) => x - y);
                const take = ds.slice(0, w.count);
                if (take.length) nearestN.push(take.reduce((x, y) => x + y, 0) / take.length);
            }

            let miss = 0;
            if (hasCount) {
                miss = w.countMode === 'atleast'
                    ? Math.max(0, w.count - inWindow)
                    : Math.abs(inWindow - w.count);
            } else if (dmax !== null) {
                miss = Math.max(0, best - dmax);
            }
            if (miss > 1e-3) { violations++; worst = Math.max(worst, miss); }
        }
        // A broken floor is a violation of the line whether or not the count
        // happened to come out right.
        if (floorBreaks) { violations += floorBreaks; worst = Math.max(worst, 1); }

        const meanCount = counts.length ? counts.reduce((x, y) => x + y, 0) / counts.length : null;
        checks.push({ a: A, b: B, dmin, dmax,
                      count: hasCount ? w.count : null, countMode: w.countMode || null,
                      n: nearest.length, violations, worst, floorBreaks,
                      closest: Number.isFinite(closest) ? closest : null,
                      meanCount, counts,
                      // Mean distance of the N nearest partners, averaged over
                      // every A atom. This is the number that says HOW unmet.
                      meanNearestN: nearestN.length
                          ? nearestN.reduce((x, y) => x + y, 0) / nearestN.length : null,
                      mean: nearest.length ? nearest.reduce((x, y) => x + y, 0) / nearest.length : null });
    }

    return { shortest: Number.isFinite(shortest) ? shortest : null, shortestPair,
             atomsInCell: inCell.length, checks,
             impossible, worstOverlap, impossiblePair,
             violations: checks.reduce((n, c) => n + (c.violations || 0), 0) };
}

/* ------------------------------------------------------------------
   Structure quality
   ------------------------------------------------------------------ */

/**
 * Bond-valence parameters R0 for cation-oxygen pairs, in Angstrom, with the
 * universal b = 0.37. From Brese & O'Keeffe (1991).
 *
 * PARTIAL AND INDICATIVE. Only oxygen anions, only common cations, and the
 * values are quoted from the literature rather than derived here - treat a
 * bond-valence sum as a strong hint that a site is right or wrong, not as a
 * measurement. Pairs not listed report nothing rather than guessing, because a
 * plausible-looking valence computed from a wrong R0 is worse than none.
 */
const BV_R0_OXYGEN = {
    'NA:1': 1.803, 'K:1': 2.132, 'MG:2': 1.693, 'CA:2': 1.967, 'SR:2': 2.118, 'BA:2': 2.285,
    'AL:3': 1.651, 'SI:4': 1.624, 'P:5': 1.617, 'S:6': 1.624, 'B:3': 1.371, 'C:4': 1.390,
    'TI:4': 1.815, 'V:5': 1.803, 'CR:3': 1.724, 'MN:2': 1.790, 'FE:2': 1.734, 'FE:3': 1.759,
    'CO:2': 1.692, 'NI:2': 1.654, 'CU:2': 1.679, 'ZN:2': 1.704, 'Y:3': 2.019, 'ZR:4': 1.928,
    'NB:5': 1.911, 'MO:6': 1.907, 'W:6': 1.921, 'LA:3': 2.172, 'PB:2': 2.112, 'BI:3': 2.094
};
const BV_B = CT_DEFAULTS.bondValenceB;

/**
 * Coordination, bond-length regularity and bond valence for one site.
 *
 * Three quantities, and the middle one is the most useful because it costs
 * nothing to compute and assumes nothing at all:
 *
 *   coordination - neighbours of the anion element within `cutoff`.
 *   spread       - the standard deviation of those bond lengths. A real
 *                  sulfate has four S-O bonds within about 0.02 A of each
 *                  other; a wrong model that happens to correlate well almost
 *                  never does. No parameters, no tables, no assumptions about
 *                  oxidation state - just the observation that coordination
 *                  polyhedra are regular.
 *   valence      - the bond-valence sum, if R0 is known for the pair. Reported
 *                  against the oxidation state that fits best, since the state
 *                  that makes the sum come out near its own charge is itself
 *                  evidence the site is right.
 */
function siteQuality(siteIdx, sites, symOps, orth, options = {}) {
    const anion = (options.anion || 'O').toUpperCase();
    const cutoff = options.cutoff ?? CT_DEFAULTS.qualityCutoffA;
    const el = String(sites[siteIdx].element).toUpperCase();

   
    const all = contactsForSite(siteIdx, sites, symOps, orth, { cutoff, max: 200 })
                    .filter(n => String(n.element).toUpperCase() === anion);
    if (!all.length) return { element: el, coordination: 0, mean: null, spread: null,
                              valence: null, shellCut: null };

    // The first coordination shell, not everything inside a fixed radius.
    //
    // A flat 3.2 A cut-off counts the four oxygens of a sulfate AND whatever
    // happens to lie beyond them, so a sulfur came back with a coordination of
    // nine and a mean bond length of 2.7 A - neither of which describes a
    // sulfate. Cutting at 1.35 times the SHORTEST contact adapts to the site
    // instead: it keeps four oxygens around a sulfur at 1.48 A and eight or
    // nine around lead at 2.6 A, which is what those coordinations are.
    //
    // Bond valence still sums over everything out to the full radius, because
    // the exponential makes distant neighbours contribute almost nothing and
    // truncating the sum at the first shell would bias every site low.
    const SHELL_FACTOR = options.shellFactor ?? CT_DEFAULTS.shellFactor;
    const dsAll = all.map(n => n.d);
    const shellCut = dsAll[0] * SHELL_FACTOR;
    const ds = dsAll.filter(d => d <= shellCut);
    const mean = ds.reduce((a, b) => a + b, 0) / ds.length;
    const spread = Math.sqrt(ds.reduce((a, d) => a + (d - mean) * (d - mean), 0) / ds.length);

    // Bond valence over the states this element has parameters for; the best
    // match is the one whose sum lands nearest its own formal charge.
    let best = null;
    const dsBV = dsAll;
    for (const key of Object.keys(BV_R0_OXYGEN)) {
        const [sym, st] = key.split(':');
        if (sym !== el) continue;
        const R0 = BV_R0_OXYGEN[key], state = parseInt(st, 10);
        const sum = dsBV.reduce((a, d) => a + Math.exp((R0 - d) / BV_B), 0);
        const dev = Math.abs(sum - state);
        if (!best || dev < best.deviation) best = { state, sum, deviation: dev };
    }

    return { element: el, coordination: ds.length, mean, spread, shellCut,
             shortest: ds[0], longest: ds[ds.length - 1],
             beyondShell: dsAll.length - ds.length, valence: best };
}

/**
 * Crystallographic residual on structure factors.
 *
 *     R = sum| |Fo| - k|Fc| | / sum|Fo|,   k from least squares
 *
 * Worth having next to the correlation because it is the number every
 * crystallographer reads, and because it weights differently: the correlation
 * is dominated by the strong reflections, while R_F counts a weak reflection
 * predicted strongly as heavily as the reverse. Two models a hundredth apart in
 * CC are often several percent apart in R.
 *
 * @param rows  [{h,k,l,Fo2,mult,d}] as prepared by normaliseObservations
 * @param opts.formFactor  (element, s) -> f; falls back to Z via opts.zOf
 */
/**
 * |Fc| for every row, from the symmetry-expanded structure.
 *
 * Split out of structureRFactor because the R factor is not the only thing
 * that wants these: comparing candidates on a powder pattern has to sum
 * |Fc|^2 over the reflections that overlap into one measured line, which
 * needs the individual values and not a residual computed from them.
 *
 * @returns {{Fo: Float64Array, Fc: Float64Array}} both on the |F| scale,
 *          UNSCALED - the scale factor belongs to whatever residual is being
 *          formed, since it depends on how the terms are grouped and weighted.
 */
function structureFactors(sites, symOps, rows, opts = {}) {
    const ff = opts.formFactor || null;
    const zOf = opts.zOf || (() => 1);
    const B = opts.overallB || 0;
    const { inCell } = expandForContacts(sites, symOps, 0);

    const Fo = new Float64Array(rows.length), Fc = new Float64Array(rows.length);
    // The form factor is a function of s alone, so it is evaluated once per
    // (element, reflection) rather than once per atom: with 24 atoms of 3
    // elements that is an eightfold saving on the inner loop, which matters
    // now that this runs for every candidate rather than only the selected one.
    const elems = [...new Set(inCell.map(a => a.element))];
    const fOf = new Map();
    for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        const s = 1 / r.d, dw = B > 0 ? Math.exp(-B * s * s / 4) : 1;
        for (const e of elems) {
            let f = ff ? ff(e, s) : null;
            if (!Number.isFinite(f)) f = zOf(e);
            fOf.set(e, f * dw);
        }
        let re = 0, im = 0;
        for (const a of inCell) {
            const f = fOf.get(a.element);
            const p = 2 * Math.PI * (r.h * a.x + r.k * a.y + r.l * a.z);
            re += f * Math.cos(p); im += f * Math.sin(p);
        }
        Fc[i] = Math.sqrt(re * re + im * im);
        Fo[i] = Math.sqrt(Math.max(0, r.Fo2));
    }
    return { Fo, Fc };
}

function structureRFactor(sites, symOps, rows, opts = {}) {
    const { Fo, Fc } = structureFactors(sites, symOps, rows, opts);
    let sFoFc = 0, sFc2 = 0;
    for (let i = 0; i < rows.length; i++) { sFoFc += Fo[i] * Fc[i]; sFc2 += Fc[i] * Fc[i]; }
    if (!(sFc2 > 0)) return null;
    const k = sFoFc / sFc2;
    let num = 0, den = 0;
    for (let i = 0; i < rows.length; i++) { num += Math.abs(Fo[i] - k * Fc[i]); den += Fo[i]; }
    return den > 0 ? num / den : null;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SHARKO_CONTACTS_VERSION, CT_DEFAULTS, expandForContacts, contactsForSite,
                       contactSummary, cartDistance, siteQuality, structureFactors, structureRFactor,
                       BV_R0_OXYGEN };
}
