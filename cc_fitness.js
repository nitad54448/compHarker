/* ------------------------------------------------------------------
   Reflection preparation for the map-correlation fitness.

   The deciding comparison is between the observed and calculated Patterson
   maps. By Parseval that is exactly the correlation of the two intensity
   lists - verified numerically to six decimal places - so the map is never
   built. Roughly 500 reflections replace 262 144 voxels, and the reciprocal
   -space form is also strictly MORE discriminating, for a reason worth
   stating plainly:

     A flat intensity I(h) = c transforms to a delta at the origin. So
     subtracting the mean intensity removes the Patterson origin peak
     EXACTLY - which is what the Pearson correlation does by construction.
     No real-space mask can do that. Measured on a scrambled structure:

        CC of the raw maps                 0.89   (useless)
        CC of the maps, origin masked      0.47   (mediocre)
        CC of the intensities, Pearson    -0.01   (correct)

   Three further things fall out of working in reciprocal space:

     - Powder overlap is handled honestly. Pawley's partitioning of intensity
       between overlapped reflections is arbitrary; grouping those lines and
       comparing GROUP SUMS compares only what was actually measured. Building
       the map bakes the arbitrary split in and you cannot undo it.

     - Resolution ramping is one integer. Groups are sorted by d*, so
       restricting the search to low-order data early - when the |F|^2
       landscape is smoothest - is a single bound on the group index.

     - R_F comes free from the same F_calc at the end.
   ------------------------------------------------------------------ */

/**
 * Orbit size of a reflection under the Laue group.
 *
 * The observed powder line is the sum over the whole orbit, and |F| is
 * constant across it, so the calculated line is m_h * |F(h)|^2. Computing m_h
 * here means the kernel evaluates ONE structure factor per orbit instead of
 * order_p of them - an 8x saving in Pnma, 48x in Ia-3.
 *
 * Friedel's law is included: F(-h) = conj(F(h)) for real scattering, so
 * (h,k,l) and (-h,-k,-l) are one line whether or not the group is centric.
 */
function reflectionMultiplicity(h, k, l, rotations) {
    const seen = new Set();
    for (const r of rotations) {
        const H = h * r[0] + k * r[3] + l * r[6];
        const K = h * r[1] + k * r[4] + l * r[7];
        const L = h * r[2] + k * r[5] + l * r[8];
        seen.add(`${H},${K},${L}`);
        seen.add(`${-H},${-K},${-L}`);
    }
    return seen.size;
}

/**
 * Groups reflections that a powder pattern cannot separate.
 *
 * Two lines whose d-spacings differ by less than a fraction of the peak width
 * are one observation. Pawley reports them separately, but the split between
 * them is a fitting artefact rather than a measurement, so the fitness should
 * compare the sum it actually observed.
 *
 * `tol` is a RELATIVE tolerance in d. 0.002 is a reasonable default for
 * laboratory data; sharper synchrotron patterns want less.
 */
function groupOverlaps(entries, tol) {
    const sorted = [...entries].sort((a, b) => a.dstar - b.dstar);
    const groups = [];
    let cur = null;
    for (const e of sorted) {
        // Compare in d, not d*, because the peak width is roughly constant in
        // 2-theta and it is d-spacing separation that decides whether two
        // lines are resolved.
        const d = 1 / e.dstar;
        if (cur && Math.abs(d - cur.d) / cur.d < tol) {
            cur.members.push(e);
            cur.iobs += e.intensity;
            cur.dstar = Math.max(cur.dstar, e.dstar);
        } else {
            cur = { members: [e], iobs: e.intensity, dstar: e.dstar, d };
            groups.push(cur);
        }
    }
    return groups;
}

/**
 * Scattering factor table: f for every (reflection, element type) pair.
 *
 * THIS REPLACES A MISTAKE. The first version normalised the OBSERVED
 * intensities shell by shell, to make them comparable with point-atom
 * calculated ones. That is backwards, and measurably so: rescaling the data
 * per shell while leaving the model unscaled means the true structure cannot
 * reach a correlation of 1. Measured on an exact structure it scored 0.88.
 *
 * The resolution fall-off belongs to the model, not the data. Putting f(s) and
 * the thermal factor on the calculated side leaves the observations untouched,
 * and the true structure then scores exactly 1.000.
 *
 * `formFactor(element, s)` supplies f at s = 1/d. The default is point atoms
 * (f = Z), which is correct for a synthetic test and adequate for a first pass
 * on real data, but for real Pawley intensities you want proper values - and
 * cctbx already has them. Adding this to cctbx_Harko_v1.py writes a table you
 * can load here rather than embedding coefficients by hand:
 *
 *     from cctbx.eltbx import xray_scattering
 *     g = xray_scattering.wk1995(element).fetch()
 *     f = g.at_stol(s / 2.0)
 *
 * `overallB` applies exp(-B s^2 / 4) on top. It is a single number and a wrong
 * one is a smooth monotonic reweighting by resolution: it lowers the peak
 * correlation but barely moves where the peak is, so it is not worth refining
 * during the search.
 */
function buildScatteringTable(refl, demand, options = {}) {
    const overallB = options.overallB ?? 0;
    const ff = options.formFactor || ((element, s) => null);
    const nElem = demand.length;
    const table = new Float32Array(refl.nRefl * nElem);

    // s per reflection, recovered from the group each belongs to. Members of a
    // group are unresolved in the pattern, so they share its d to within the
    // overlap tolerance and one value serves them all.
    const sOf = new Float32Array(refl.nRefl);
    for (let g = 0; g < refl.nGroups; g++) {
        const st = refl.groupStart[g], c = refl.groupCount[g];
        for (let m = 0; m < c; m++) sOf[st + m] = refl.groupDstar[g];
    }

    let usedPoint = false;
    for (let r = 0; r < refl.nRefl; r++) {
        const s = sOf[r];
        const dw = overallB > 0 ? Math.exp(-overallB * s * s / 4) : 1;
        for (let e = 0; e < nElem; e++) {
            let f = ff(demand[e].element, s);
            if (!Number.isFinite(f)) { f = demand[e].z; usedPoint = true; }
            table[r * nElem + e] = f * dw;
        }
    }
    return { table, nElem, usedPointAtoms: usedPoint };
}


/**
 * Everything the CC kernel needs, from the Pawley reflection list.
 *
 * IMPORTANT: pass the UNIQUE reflections with their full line intensities -
 * crystalData.reflections as read from the file. Do NOT pass the output of
 * expandReflections(), which divides each line over its orbit for the map;
 * here the orbit is accounted for by the multiplicity instead.
 *
 * @returns {Object} buffers plus the ramp schedule
 */
function buildReflectionSet(reflections, cell, rotations, options = {}) {
    const overlapTol = options.overlapTol ?? 0.002;
    const B = sharkoReciprocalMatrix(cell);

    const entries = [];
    const problems = [];
    for (const r of reflections) {
        const h = r.h | 0, k = r.k | 0, l = r.l | 0;
        if (!h && !k && !l) continue;                 // F(000) is not measured
        const I = Number(r.intensity);
        if (!Number.isFinite(I)) continue;
        if (Math.abs(h) > 511 || Math.abs(k) > 511 || Math.abs(l) > 511) {
            problems.push(`Reflection ${h} ${k} ${l} exceeds the +/-511 packing range.`);
            continue;
        }
        entries.push({ h, k, l, intensity: I,
                       dstar: sharkoDStar(h, k, l, B),
                       mult: reflectionMultiplicity(h, k, l, rotations) });
    }
    if (!entries.length) throw new Error('No usable reflections for the correlation fitness.');

    const groups = groupOverlaps(entries, overlapTol);
    // Observations are used exactly as measured. See buildScatteringTable()
    // for why any per-shell rescaling of the data is a mistake.
    for (const g of groups) g.iobsNorm = g.iobs;

    // Groups are already in ascending d*, so a resolution ramp is a single
    // bound on the group index and needs no reordering at run time.
    const nGroups = groups.length;
    const nRefl = entries.length;

    const reflPack = new Uint32Array(nRefl * 2);   // [hkl packed, multiplicity]
    const groupStart = new Uint32Array(nGroups);
    const groupCount = new Uint32Array(nGroups);
    const groupIobs = new Float32Array(nGroups);
    const groupDstar = new Float32Array(nGroups);

    let w = 0;
    groups.forEach((g, gi) => {
        groupStart[gi] = w;
        groupCount[gi] = g.members.length;
        groupIobs[gi] = g.iobsNorm;
        groupDstar[gi] = g.dstar;
        for (const e of g.members) {
            // 10 bits per index, biased by 512 so negatives survive.
            reflPack[w * 2] = ((e.h + 512) & 0x3FF)
                            | (((e.k + 512) & 0x3FF) << 10)
                            | (((e.l + 512) & 0x3FF) << 20);
            reflPack[w * 2 + 1] = e.mult;
            w++;
        }
    });

    const overlapped = groups.filter(g => g.members.length > 1).length;

    return {
        reflPack, groupStart, groupCount, groupIobs, groupDstar,
        nRefl, nGroups, overlapped, problems,
        dMin: 1 / groups[nGroups - 1].dstar,
        dMax: 1 / groups[0].dstar,
    };
}

/**
 * Resolution ramp: how many groups are active at a given point in the run.
 *
 * The |F|^2 landscape is far more oscillatory than the Patterson vector sum,
 * so a swarm turned loose on the full reflection set tends to stall in a local
 * maximum. Starting with the low-order data gives a smooth surface with few
 * maxima, and adding shells as the run proceeds sharpens it around whatever
 * basin the swarm has already found. It is also cheaper early, when most
 * particles are going to be discarded anyway.
 *
 * Returns the group count to use, from `startFrac` of the data at generation 0
 * to all of it at `fullBy`.
 */
function rampedGroupCount(nGroups, generation, maxGen, startFrac = 0.25, fullBy = 0.6) {
    const t = Math.min(1, generation / Math.max(1, maxGen * fullBy));
    const frac = startFrac + (1 - startFrac) * t;
    return Math.max(8, Math.min(nGroups, Math.round(nGroups * frac)));
}

/**
 * Reference CC on the CPU: the same quantity the kernel computes.
 *
 * Used to validate the kernel and to score a final structure without a GPU
 * round trip. `atoms` are the full cell contents as {x, y, z, z: atomic number}.
 */
function correlationCPU(atoms, refl, nGroupsActive, ftab) {
    const n = Math.min(nGroupsActive ?? refl.nGroups, refl.nGroups);
    let sIo = 0, sIo2 = 0, sIc = 0, sIc2 = 0, sIoIc = 0;
    for (let g = 0; g < n; g++) {
        let ic = 0;
        const s0 = refl.groupStart[g], c = refl.groupCount[g];
        for (let m = 0; m < c; m++) {
            const pk = refl.reflPack[(s0 + m) * 2];
            const h = (pk & 0x3FF) - 512;
            const k = ((pk >> 10) & 0x3FF) - 512;
            const l = ((pk >> 20) & 0x3FF) - 512;
            const mult = refl.reflPack[(s0 + m) * 2 + 1];
            let re = 0, im = 0;
            for (const a of atoms) {
                const f = ftab ? ftab.table[(s0 + m) * ftab.nElem + a.type] : a.zn;
                const p = 2 * Math.PI * (h * a.x + k * a.y + l * a.z);
                re += f * Math.cos(p); im += f * Math.sin(p);
            }
            ic += mult * (re * re + im * im);
        }
        const io = refl.groupIobs[g];
        sIo += io; sIo2 += io * io;
        sIc += ic; sIc2 += ic * ic;
        sIoIc += io * ic;
    }
    const num = n * sIoIc - sIo * sIc;
    const den = Math.sqrt(Math.max(0, n * sIo2 - sIo * sIo) * Math.max(0, n * sIc2 - sIc * sIc));
    return den > 1e-12 ? num / den : 0;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { reflectionMultiplicity, groupOverlaps, buildScatteringTable,
                       buildReflectionSet, rampedGroupCount, correlationCPU };
}
