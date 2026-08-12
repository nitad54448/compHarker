// Bumped whenever this file changes. Harko.html compares it against what it
// expects and says so if a browser cache is serving something older - a stale
// module reports errors at line numbers that no longer exist, which sends you
// looking for a bug that was already fixed.
const SHARKO_WYCKOFF_ASSIGN_VERSION = '2026-08-09j';

/* ------------------------------------------------------------------
   Wyckoff-space search for sHarko.

   The swarm currently parametrises every independent atom as three free
   coordinates in a general position, so the cell contents it produces are
   whatever the symmetry happens to generate - Pb8S8O24 rather than PbSO4.
   Composition is checked after the fact and cannot be imposed.

   This module inverts that. A trial structure is described by

       (Wyckoff assignment) x (free parameters of that assignment)

   where an assignment picks, for each element, a multiset of Wyckoff
   positions whose multiplicities sum to the required number of atoms per
   cell. Every structure the swarm can then express is stoichiometric by
   construction, and the dimensionality drops: a site on 4c (x,1/4,z) has two
   parameters, one on 4a (0,0,0) has none.

   Three things are exported:

     enumerateAssignments()  the combinatorics, with pruning
     rankAssignments()       a cheap prior, using Harker evidence when present
     buildAssignmentTables() flattening for the multi-assignment GPU kernel

   plus projectSites(), which is what keeps the PSO inside the subspace.
   ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Wyckoff position helpers                                           */
/* ------------------------------------------------------------------ */

/**
 * Degrees of freedom of a position.
 *
 * Uses the generator's exact rank(P). There is deliberately NO string fallback:
 * cctbx does not always print a special operator in the tidy form one expects.
 * The projector for (x,x,x) in Ia-3 comes out as
 *
 *     1/3*x+1/3*y+1/3*z, 1/3*x+1/3*y+1/3*z, 1/3*x+1/3*y+1/3*z
 *
 * which mentions all three symbols but has rank 1. Counting symbols would call
 * that three free parameters, giving the site three search dimensions where it
 * has one - and the excess two would be a flat direction the swarm wastes its
 * time in, with nothing to signal the mistake. Better to fail loudly: if
 * n_free is missing the database was built by an older generator, and the
 * answer is to re-run it rather than to guess.
 */
function wyckoffFreedom(w) {
    if (Number.isFinite(w.n_free)) return w.n_free;
    throw new Error(
        `Wyckoff position ${w.multiplicity || ''}${w.letter || '?'} has no n_free. ` +
        `Regenerate the space-group database with cctbx_Harko_v1.py - the degrees ` +
        `of freedom cannot be recovered reliably from the special_op string.`);
}

/** Projection matrix P and offset T as plain floats, cached on the position. */
function wyckoffProjector(w) {
    if (w._proj) return w._proj;
    let P, T;
    if (Array.isArray(w.P_num) && w.P_num.length === 9 && w.P_den) {
        P = w.P_num.map(v => v / w.P_den);
        T = (w.T_num || [0, 0, 0]).map(v => v / (w.T_den || 1));
    } else {
        // No matrix in the database: treat as a general position. This is the
        // honest fallback - it constrains nothing rather than constraining
        // something wrong - but the multiplicity will then be whatever the
        // group generates, so the caller should warn.
        P = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        T = [0, 0, 0];
        w._degraded = true;
    }
    w._proj = { P, T };
    return w._proj;
}

/**
 * Projects a fractional point onto a Wyckoff position, writing into `out`.
 *
 * special_op is idempotent, so this is a true projection: applying it to a
 * point already on the position leaves it alone. That is what allows the PSO
 * update to run untouched - see projectSites().
 */
function projectOnto(w, x, y, z, out) {
    const { P, T } = wyckoffProjector(w);
    out[0] = P[0] * x + P[1] * y + P[2] * z + T[0];
    out[1] = P[3] * x + P[4] * y + P[5] * z + T[1];
    out[2] = P[6] * x + P[7] * y + P[8] * z + T[2];
    return out;
}

const _tmp3 = new Float64Array(3);

/**
 * How far a point has to move, in Angstrom, to sit on a Wyckoff position.
 *
 * This is the quantity that turns Harker output into evidence about
 * assignments: a consolidated site 0.04 A from 4c but 1.8 A from 4a says the
 * heavy atom is on 4c, before any swarm has run.
 */
function wyckoffShift(w, p, metric) {
    projectOnto(w, p.x, p.y, p.z, _tmp3);
    let dx = _tmp3[0] - p.x, dy = _tmp3[1] - p.y, dz = _tmp3[2] - p.z;

    // `metric` is either a bare fractional->Cartesian matrix or a reduction
    // from sharkoReducedCell(). With the reduction, the rounding below happens
    // in the reduced basis, where it really is the nearest image; with a bare
    // matrix it is the old convention, which is only right for a near-
    // orthogonal cell. The shift being measured is small, so the difference
    // rarely bites - but this feeds the pre-search ranking, and an assignment
    // wrongly reported as far from a Harker peak is one that gets searched last
    // or not at all.
    const red = (metric && metric.xform) ? metric : null;
    const orth = red ? red.orth : metric;
    if (red) {
        const X = red.xform;
        const u = X[0] * dx + X[1] * dy + X[2] * dz;
        const v = X[3] * dx + X[4] * dy + X[5] * dz;
        const t = X[6] * dx + X[7] * dy + X[8] * dz;
        dx = u; dy = v; dz = t;
    }
    dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);

    const cx = orth[0] * dx + orth[1] * dy + orth[2] * dz;
    const cy = orth[3] * dx + orth[4] * dy + orth[5] * dz;
    const cz = orth[6] * dx + orth[7] * dy + orth[8] * dz;
    return Math.sqrt(cx * cx + cy * cy + cz * cz);
}

/* ------------------------------------------------------------------ */
/*  Enumeration                                                        */
/* ------------------------------------------------------------------ */

/**
 * Multisets of positions whose multiplicities sum to `target`.
 *
 * Indices are non-decreasing, which enumerates each multiset once instead of
 * once per ordering. A position with no free parameter is a single point in
 * the cell, so it can be used at most once; one with parameters can hold
 * several independent atoms at different x (PbSO4 has two distinct oxygens on
 * 4c), capped by maxRepeat to keep the space finite.
 */
function multisetsSummingTo(positions, target, opts) {
    const out = [];
    const counts = new Int32Array(positions.length);
    const chosen = [];

    const rec = (start, remaining) => {
        if (remaining === 0) { out.push(chosen.slice()); return; }
        if (chosen.length >= opts.maxSites) return;
        if (out.length >= opts.maxPerElement) return;
        for (let i = start; i < positions.length; i++) {
            const m = positions[i].multiplicity;
            if (!m || m > remaining) continue;
            const cap = wyckoffFreedom(positions[i]) > 0 ? opts.maxRepeat : 1;
            if (counts[i] >= cap) continue;
            counts[i]++; chosen.push(i);
            rec(i, remaining - m);
            chosen.pop(); counts[i]--;
            if (out.length >= opts.maxPerElement) return;
        }
    };
    rec(0, target);
    return out;
}

/**
 * All Wyckoff assignments consistent with a composition.
 *
 * @param {Array}  wyckoff  the setting's wyckoff array from the JSON
 * @param {Array}  demand   [{ element, z, r, count }] - count is atoms PER
 *                          CELL, i.e. Z x (atoms per formula unit)
 * @param {Object} options  maxSites, maxRepeat, maxPerElement, maxTotal
 * @returns {Array} [{ sites: [{ element, elementIdx, z, r, w, wIdx }], nSites }]
 *
 * The only cross-element constraint is that a zero-parameter position is one
 * point in the cell and cannot be shared: if Pb takes 4a, no other element
 * may. Positions with parameters are freely reusable.
 */
function enumerateAssignments(wyckoff, demand, options = {}) {
    // Caps that cannot possibly be met are not caps, they are a wall.
    //
    // Both defaults used to be a flat 4, and 4 is generous in a high-symmetry
    // group and impossible in a low one. PbSO4 as Pb4S4O16 in P-1 needs eight
    // oxygens on 2i - eight distinct sites, eight reuses of one position - and
    // the enumeration returned nothing with a message blaming Z, which was the
    // one thing that was right. The floor below is what the composition
    // arithmetically requires; the caller's value is honoured when it is
    // larger, and `ceiling` is what stops a genuinely absurd request.
    const ceiling = options.ceiling ?? 24;
    const mults = wyckoff.map(w => w.multiplicity).filter(m => m > 0);
    const mMax = mults.length ? Math.max(...mults) : 1;
    const maxCount = demand.reduce((n, d) => Math.max(n, d.count), 0);
    const needed = Math.min(ceiling, Math.ceil(maxCount / mMax));

    const opts = {
        ...options,
        // distinct sites one element may occupy
        maxSites: Math.min(ceiling, Math.max(options.maxSites ?? 4, needed)),
        // times one position may be reused
        maxRepeat: Math.min(ceiling, Math.max(options.maxRepeat ?? 4, needed)),
        maxPerElement: options.maxPerElement ?? 500,
        maxTotal: options.maxTotal ?? 4000
    };

    // Largest multiplicity first: the recursion then hits complete solutions
    // early and the maxPerElement cap truncates the tail of many-small-sites
    // combinations rather than the head of plausible ones.
    const idx = wyckoff.map((w, i) => i)
                       .sort((a, b) => wyckoff[b].multiplicity - wyckoff[a].multiplicity);
    const sorted = idx.map(i => wyckoff[i]);

    const perElement = demand.map(d => {
        const sets = multisetsSummingTo(sorted, d.count, opts);
        return sets.map(s => s.map(k => idx[k]));   // back to original indices
    });

    for (let e = 0; e < demand.length; e++) {
        if (perElement[e].length === 0) {
            // Say WHICH constraint failed. "Check Z" was the old advice and it
            // was usually wrong: the multiplicities available almost always can
            // sum to the demand, and what actually stopped the recursion was a
            // cap on how many sites were allowed to do it.
            const avail = [...new Set(mults)].sort((a, b) => a - b);
            const g = avail.reduce((x, y) => { while (y) { [x, y] = [y, x % y]; } return x; }, 0);
            const reachable = demand[e].count % g === 0;
            const minSites = Math.ceil(demand[e].count / mMax);
            return { assignments: [], error:
                `No Wyckoff assignment gives ${demand[e].count} ${demand[e].element} atoms ` +
                `per cell. ` +
                (reachable
                    ? `The multiplicities available (${avail.join(', ')}) can reach that total, ` +
                      `but only with at least ${minSites} sites, and the search is capped at ` +
                      `${opts.maxSites} distinct site(s) per element and ${opts.maxRepeat} reuse(s) ` +
                      `of any one position. Raise the caps, or check Z - the formula is multiplied ` +
                      `by Z, so writing the cell contents as the formula AND setting Z above 1 ` +
                      `counts every atom twice.`
                    : `The multiplicities available (${avail.join(', ')}) share a common factor of ` +
                      `${g}, so no combination of them can sum to ${demand[e].count}. Check Z.`) };
        }
    }

    const fixedOnly = new Set();
    wyckoff.forEach((w, i) => { if (wyckoffFreedom(w) === 0) fixedOnly.add(i); });

    const assignments = [];
    const cur = new Array(demand.length);
    const rec = (e, usedFixed) => {
        if (assignments.length >= opts.maxTotal) return;
        if (e === demand.length) {
            const sites = [];
            for (let k = 0; k < demand.length; k++) {
                for (const wIdx of cur[k]) {
                    sites.push({
                        element: demand[k].element, elementIdx: k,
                        z: demand[k].z, r: demand[k].r,
                        wIdx, w: wyckoff[wIdx]
                    });
                }
            }
            assignments.push({ sites, nSites: sites.length });
            return;
        }
        for (const combo of perElement[e]) {
            // A fixed point can host one atom, full stop.
            let clash = false;
            const added = [];
            for (const wIdx of combo) {
                if (!fixedOnly.has(wIdx)) continue;
                if (usedFixed.has(wIdx)) { clash = true; break; }
                usedFixed.add(wIdx); added.push(wIdx);
            }
            if (!clash) { cur[e] = combo; rec(e + 1, usedFixed); }
            for (const wIdx of added) usedFixed.delete(wIdx);
            if (assignments.length >= opts.maxTotal) return;
        }
    };
    rec(0, new Set());

    return { assignments, truncated: assignments.length >= opts.maxTotal };
}

/* ------------------------------------------------------------------ */
/*  Ranking                                                            */
/* ------------------------------------------------------------------ */

/**
 * Orders assignments best-first on evidence available before any search.
 *
 * Two terms:
 *
 *   Harker residual. Every consolidated Harker site is a candidate position
 *   for a heavy atom. For each heavy site in the assignment, the distance to
 *   the nearest candidate that also lies ON the assigned position is a direct
 *   measure of whether that assignment is compatible with the Patterson we
 *   already computed. This is far more informative than any chemical rule of
 *   thumb and it is free - no GPU work at all.
 *
 *   Parsimony. Among assignments the data cannot distinguish, prefer fewer
 *   distinct sites. This is a tiebreaker, weighted low enough that it never
 *   overrides evidence.
 */
function rankAssignments(assignments, ctx) {
    const { orth, harkerSites = [], heavyZ = 0, parsimony = 0.15 } = ctx;

    for (const A of assignments) {
        let evidence = 0, n = 0;
        for (const s of A.sites) {
            if (s.z < heavyZ) continue;
            let best = Infinity;
            for (const h of harkerSites) {
                const d = wyckoffShift(s.w, h, orth);
                if (d < best) best = d;
            }
            if (best !== Infinity) { evidence += Math.min(best, 3.0); n++; }
        }
        A.harkerResidual = n ? evidence / n : null;
        A.priorScore = (n ? evidence / n : 1.5) + parsimony * A.nSites;
    }
    assignments.sort((a, b) => a.priorScore - b.priorScore);
    return assignments;
}

/* ------------------------------------------------------------------ */
/*  Flattening for the GPU                                             */
/* ------------------------------------------------------------------ */

/**
 * Packs every assignment into the flat buffers swarm_cc.wgsl expects.
 *
 * The crucial invariant: the total number of atoms generated in the cell is
 * the same for every assignment, because the composition is fixed. That makes
 * the generated-atom arrays a constant size, removes the need for the shader's
 * distance-based special-position collapse entirely, and - because the pair
 * weight sum is then identical everywhere - makes raw fitness directly
 * comparable BETWEEN assignments. Comparing them is the whole point.
 *
 * Per generated atom g of assignment A, at index A*nTot + g:
 *   genSite  which site of A it belongs to (indexes the particle's coords)
 *   genOp    which symmetry operator produced it
 *   genZ     atomic number
 *   genType  element index, for the distance restraints
 *
 * genOp comes from the position's precomputed coset operators, so exactly
 * `multiplicity` distinct atoms are produced with no tolerance to tune.
 */
function buildAssignmentTables(assignments, symOpCount, demand) {
    const nTot = demand.reduce((n, d) => n + d.count, 0);
    const nA = assignments.length;
    const maxSites = assignments.reduce((m, A) => Math.max(m, A.nSites), 0);

    const genSite = new Uint32Array(nA * nTot);
    const genOp = new Uint32Array(nA * nTot);
    const genZ = new Float32Array(nA * nTot);
    const genType = new Uint32Array(nA * nTot);
    const siteProj = new Float32Array(nA * maxSites * 12);   // P(9) + T(3)
    const siteCount = new Uint32Array(nA);
    const warnings = [];

    assignments.forEach((A, ai) => {
        let g = 0;
        A.sites.forEach((s, si) => {
            const cosets = Array.isArray(s.w.coset_ops) && s.w.coset_ops.length
                ? s.w.coset_ops
                : null;
            if (!cosets) {
                warnings.push(`Wyckoff ${s.w.letter} has no coset_ops in the database - ` +
                              `re-run the generator with the Wyckoff patch.`);
            }
            const ops = cosets || Array.from({ length: s.w.multiplicity }, (_, k) => k % symOpCount);
            for (const op of ops) {
                if (g >= nTot) break;
                genSite[ai * nTot + g] = si;
                genOp[ai * nTot + g] = op;
                genZ[ai * nTot + g] = s.z;
                genType[ai * nTot + g] = s.elementIdx;
                g++;
            }
            const { P, T } = wyckoffProjector(s.w);
            const b = (ai * maxSites + si) * 12;
            for (let k = 0; k < 9; k++) siteProj[b + k] = P[k];
            for (let k = 0; k < 3; k++) siteProj[b + 9 + k] = T[k];
        });
        // Unused site slots project to the origin so a stale coordinate can
        // never leak into a shorter assignment's atom list.
        for (let si = A.nSites; si < maxSites; si++) {
            const b = (ai * maxSites + si) * 12;
            for (let k = 0; k < 12; k++) siteProj[b + k] = 0;
        }
        siteCount[ai] = A.nSites;
        if (g !== nTot) {
            warnings.push(`Assignment ${ai} generated ${g} atoms, expected ${nTot}.`);
        }
    });

    // Constant pair weight: (sum z)^2 - sum z^2, halved. Identical for every
    // assignment because the cell contents are.
    let sz = 0, sz2 = 0;
    for (const d of demand) { sz += d.z * d.count; sz2 += d.z * d.z * d.count; }
    const normW = Math.max(1e-6, 0.5 * (sz * sz - sz2));

    // Bit-packed for the kernel: site | op << 8 | type << 20. WebGPU
    // guarantees only eight storage buffers per stage and the previous kernel
    // already used all eight, so three per-atom u32 fields have to travel as
    // one. The field widths are checked rather than assumed - 256 sites, 4096
    // operators (order_z tops out at 192) and 16 elements are all far beyond
    // any real problem, but silently truncating one would corrupt the atom
    // list in a way that looks like a bad structure rather than a bug.
    // The check THROWS. It used to push a warning and break out of the loop,
    // which left genPack filled with zeros from the offending index onward -
    // every remaining atom becomes site 0, operator 0, element 0. That is the
    // exact silent corruption the comment above says it is guarding against:
    // the run completes, the correlation is meaningless, and the only trace is
    // a line in a log nobody reads. A structure that cannot be represented is
    // a failure, not a caveat.
    const genPack = new Uint32Array(nA * nTot);
    for (let i = 0; i < genPack.length; i++) {
        const s = genSite[i], o = genOp[i], t = genType[i];
        if (s > 0xFF || o > 0xFFF || t > 0xF) {
            throw new Error(
                `Cannot pack the generated-atom table: site ${s} (limit 255), operator ${o} ` +
                `(limit 4095), element ${t} (limit 15) exceeds the kernel's field widths. ` +
                `This needs ${maxSites} sites and ${symOpCount} operators over ` +
                `${demand.length} element(s) - reduce Z or the number of elements.`);
        }
        genPack[i] = s | (o << 8) | (t << 20);
    }

    return { nTot, maxSites, genSite, genOp, genZ, genType, genPack,
             siteProj, siteCount, normW, warnings: [...new Set(warnings)] };
}

/**
 * The single f32 lookup buffer: [ zByType | rMin matrix | bond rules ].
 *
 * Concatenated for the same binding-budget reason as genPack. Offsets go in
 * the uniform block.
 *
 * rMin is a full nElem x nElem lower-bound matrix. Every pair starts at the
 * minimum-contact floor (`options.minContact`, a hard floor on every pair);
 * `windows` then overrides individual pairs:
 *
 *     [{ a: 'S', b: 'O', dmin: 1.35, dmax: 1.65 }]
 *
 * dmin is symmetric and goes into the matrix. dmax becomes a directed
 * nearest-neighbour rule, and is emitted BOTH ways round unless the caller
 * asks otherwise: "every S has an O within 1.65 A" and "every O has an S
 * within 1.65 A" are different statements, and for a sulfate both are true.
 * For a rule like Pb-O where the partner counts differ wildly, one direction
 * is usually what you want.
 */
/**
 * (elementA, elementB) -> the clash floor the search used, in Angstrom.
 */
function floorLookup(demand, rMin, nElem) {
    const idx = {};
    demand.forEach((d, i) => { idx[String(d.element).toUpperCase()] = i; });
    return (a, b) => {
        const i = idx[String(a).toUpperCase()], j = idx[String(b).toUpperCase()];
        return (i === undefined || j === undefined) ? 0 : rMin[i * nElem + j];
    };
}

function buildRestraintTables(demand, windows = [], options = {}) {
    const nElem = demand.length;
    const idxOf = {};
    demand.forEach((d, i) => { idxOf[d.element.toUpperCase()] = i; });

    // THE CLASH FLOOR, IN ONE RULE:
    //
    //     rMin(A,B) = the dmin the user stated for that pair,   if they stated one
    //               = the minimum-contact slider,               otherwise
    //
    // Nothing else feeds it. dmax does NOT - an upper bound on where partners
    // may sit says nothing about how close atoms may approach, and letting it
    // move the floor made the two ends of a constraint interact in a way no one
    // could follow from the input.
    //
    // WHY THE RADIUS RULE IS GONE. The floor used to be
    // min(slider, 0.65 x (rc_A + rc_B)), which meant the slider could only ever
    // LOOSEN a pair: raised above a pair's radius sum it did nothing at all.
    // On PbSO4 a slider set to 1.10 A still returned candidates with O-O at
    // 0.94, because the O-O radius floor is 0.86 and min() kept it. That is not
    // a defensible reading of a control called "minimum contact distance", and
    // no amount of documentation fixes a control whose name is the opposite of
    // its behaviour. One number, applied to every pair, is also the only
    // version a user can check: the Min d column either respects it or the code
    // is wrong.
    //
    // The old rule's motive was real and is preserved in the DEFAULT rather
    // than in the arithmetic. A floor is a guess and the two ways of being
    // wrong are not symmetric: too low merely fails to catch nonsense the
    // correlation would have rejected anyway, while too high makes the right
    // answer UNREACHABLE - and charges it more than any wrong one, since the
    // clash term scales with overlap depth. So the default is deliberately
    // permissive, and raising it is the user's decision to make.
    //
    // 1.0 A: below every bond that actually occurs (the shortest in ordinary
    // inorganics is a little over 1.1 A, B-O and C-O), and far above what a
    // wrong Wyckoff assignment produces - a Pb-O contact of 0.32 A correlated
    // at 0.9428 and ranked fourth in a real run when there was no floor at all.
    // An explicit 0 turns the floor off; only an ABSENT option falls back to
    // the default, so a caller that forgets to pass one is not silently left
    // with no clash term.
    const DEFAULT_CONTACT_FLOOR_A = 1.0;
    const floor = (options.minContact === undefined || options.minContact === null)
        ? DEFAULT_CONTACT_FLOOR_A
        : (Number.isFinite(options.minContact) && options.minContact > 0 ? options.minContact : 0);
    const zByType = demand.map(d => d.z);
    const rMin = new Float64Array(nElem * nElem);
    rMin.fill(floor);

    // Six floats per rule: [aType, bType, dmin, dmax, count, mode].
    //   mode 0 = no count requirement (nearest-neighbour rule only)
    //        1 = exactly `count` neighbours in the window
    //        2 = at least `count`
    // dmax is +inf when the constraint left the upper side open.
    const RULE_STRIDE = 6;
    const MODE = { none: 0, exact: 1, atleast: 2 };
    const problems = [];
    const rules = [];
    for (const w of windows) {
        const a = idxOf[String(w.a).toUpperCase()];
        const b = idxOf[String(w.b).toUpperCase()];
        if (a === undefined || b === undefined) {
            problems.push(`Constraint ${w.a}-${w.b} names an element that is not in the composition.`);
            continue;
        }
        const dmin = Number.isFinite(w.dmin) ? w.dmin : null;
        const dmax = Number.isFinite(w.dmax) ? w.dmax : null;
        if (dmin !== null && dmax !== null && dmax < dmin) {
            problems.push(`Constraint ${w.a}-${w.b} has dmax (${dmax}) below dmin (${dmin}).`);
            continue;
        }

        // dmin is a HARD FLOOR, always, whether or not a count was asked for.
        // It goes into the symmetric rMin matrix, which is what the kernel
        // charges as a clash - so "S O - 1.4/" is expressed entirely here and
        // needs no rule at all.
        // A stated dmin REPLACES the global floor for that pair - not max(),
        // not min(). Both numbers come from the user, and the more specific one
        // wins, in either direction: a constraint may hold one pair further
        // apart than the slider, or admit an approach the slider would forbid.
        // Anything else would make a constraint mean different things depending
        // on where the slider happened to be sitting.
        if (dmin !== null) {
            rMin[a * nElem + b] = dmin;
            rMin[b * nElem + a] = dmin;
        }

        const hasCount = Number.isFinite(w.count) && w.count >= 0;
        const mode = hasCount ? (w.countMode === 'atleast' ? MODE.atleast : MODE.exact) : MODE.none;

        // A rule is only needed when something beyond the floor is being asked:
        // a coordination number, or an upper distance.
        if (!hasCount && dmax === null) continue;

        const rec = [a, b, dmin ?? 0, dmax ?? Infinity, hasCount ? w.count : 0, mode];
        rules.push(rec);
        // A coordination number is DIRECTED - "four O around every S" says
        // nothing about how many S surround an O - so it is never mirrored.
        // A bare upper bound is a nearest-neighbour condition and is symmetric.
        if (!hasCount && w.bothWays !== false && a !== b) {
            rules.push([b, a, dmin ?? 0, dmax ?? Infinity, 0, MODE.none]);
        }
    }
    if (rules.length > 8) problems.push(`${rules.length} distance rules; the kernel holds 8.`);
    // The kernel keeps the N nearest partners of every counted rule so the
    // coordination penalty can be charged by distance rather than by count.
    // Those slots are a fixed private array, so the total is bounded.
    const MAX_COORD_SLOTS = 16;
    const slotsNeeded = rules.reduce((n, r) => n + (r[5] !== MODE.none ? r[4] : 0), 0);
    if (slotsNeeded > MAX_COORD_SLOTS) {
        problems.push(`The coordination numbers add up to ${slotsNeeded} neighbours; the kernel ` +
                      `tracks ${MAX_COORD_SLOTS}. Drop a constraint or lower a coordination number.`);
    }

    const rMinOff = zByType.length;
    const ruleOff = rMinOff + rMin.length;
    const tables = new Float32Array(ruleOff + rules.length * RULE_STRIDE);
    tables.set(zByType, 0);
    tables.set(rMin, rMinOff);
    // Float32Array turns Infinity into Infinity, which compares correctly in
    // WGSL; no sentinel needed.
    rules.forEach((r, k) => tables.set(r, ruleOff + k * RULE_STRIDE));

    // The longest distance any comparison in the kernel depends on. The
    // minimum-image convention only has to be exact out to here, so this is
    // what decides whether the reduced basis alone is enough or the kernel must
    // also search the neighbouring translations. See sharkoReducedCell().
    let maxDistance = 0;
    for (let i = 0; i < rMin.length; i++) maxDistance = Math.max(maxDistance, rMin[i]);
    for (const r of rules) if (Number.isFinite(r[3])) maxDistance = Math.max(maxDistance, r[3]);

    return { tables, rMinOff, ruleOff, nRules: Math.min(rules.length, 8), nElem,
             ruleStride: RULE_STRIDE, maxDistance, problems,
             // The resolved floors, so the post-search filter can reject on
             // exactly what the search enforced. They used to disagree: the
             // search applied per-pair floors while the filter applied a flat
             // 0.9 A to everything, which let the search spend a run on
             // contacts the filter then threw out. Anything reading this gets
             // the per-pair answer including any constraint override, not the
             // slider value, which is why it stays a lookup and not a number.
             floors: floorLookup(demand, rMin, nElem) };
}

/** Symmetry operators as 12 floats each: r[9] then t[3]. */
function packSymOps(symOps) {
    const out = new Float32Array(symOps.length * 12);
    symOps.forEach((op, i) => { out.set(op.r, i * 12); out.set(op.t, i * 12 + 9); });
    return out;
}

/**
 * Projects one particle's coordinates onto its assignment's subspaces.
 *
 * Call this after initialising a chain and on EVERY proposal. It is
 * idempotent, so the cost is one 3x3 multiply per site and calling it too
 * often is harmless.
 *
 * Under the Monte Carlo sampler this is not a correction for drift - it is the
 * only thing keeping a structure on its subspace. A particle swarm stayed there
 * largely for free, because its update is a linear combination of points
 * already in the affine subspace; a Gaussian proposal is not, and lands off the
 * subspace almost surely. A move is therefore a step in the ambient cell
 * followed by a projection, which is still a symmetric proposal on the subspace
 * and leaves the Metropolis acceptance rule valid.
 */
function projectSites(positions, particleIdx, assignIdx, tables) {
    const { maxSites, siteProj, siteCount } = tables;
    const nS = siteCount[assignIdx];
    const stride = maxSites * 3;
    for (let s = 0; s < nS; s++) {
        const b = (assignIdx * maxSites + s) * 12;
        const o = particleIdx * stride + s * 3;
        const x = positions[o], y = positions[o + 1], z = positions[o + 2];
        positions[o]     = siteProj[b]     * x + siteProj[b + 1] * y + siteProj[b + 2] * z + siteProj[b + 9];
        positions[o + 1] = siteProj[b + 3] * x + siteProj[b + 4] * y + siteProj[b + 5] * z + siteProj[b + 10];
        positions[o + 2] = siteProj[b + 6] * x + siteProj[b + 7] * y + siteProj[b + 8] * z + siteProj[b + 11];
        for (let k = 0; k < 3; k++) {
            const i = o + k;
            positions[i] -= Math.floor(positions[i]);
        }
    }
    // Slots this assignment does not use are parked at the origin so they can
    // never be read as coordinates.
    for (let s = nS; s < maxSites; s++) {
        const o = particleIdx * stride + s * 3;
        positions[o] = positions[o + 1] = positions[o + 2] = 0;
    }
}

/**
 * Shares a particle budget over a set of assignments, weighted by dimension.
 *
 * Split out because it is needed twice: once when a wave starts, and again
 * every time successive halving retires the losers and the survivors inherit
 * their particles. Redistributing uniformly at that second point would undo
 * the weighting exactly when the budget per assignment finally becomes large
 * enough for it to matter.
 */
function weightedAssign(ids, numParticles, minPer, weights) {
    const out = new Uint32Array(numParticles);
    if (!ids.length) return out;

    const floorEach = Math.min(minPer, Math.floor(numParticles / ids.length));
    const w = ids.map(a => Math.max(1e-6, (weights && weights[a]) || 1));
    const total = w.reduce((x, y) => x + y, 0);
    const surplus = Math.max(0, numParticles - floorEach * ids.length);
    const counts = w.map(x => floorEach + Math.floor(surplus * x / total));

    // Rounding leaves a few over; they go to the largest problems.
    const order = ids.map((_, k) => k).sort((a, b) => w[b] - w[a]);
    let assigned = counts.reduce((x, y) => x + y, 0), oi = 0;
    while (assigned < numParticles) { counts[order[oi++ % order.length]]++; assigned++; }
    while (assigned > numParticles) {
        const k = order[order.length - 1 - (oi++ % order.length)];
        if (counts[k] > 1) { counts[k]--; assigned--; }
    }

    let p = 0;
    ids.forEach((a, k) => { for (let n = 0; n < counts[k] && p < numParticles; n++) out[p++] = a; });
    while (p < numParticles) out[p++] = ids[p % ids.length];
    return out;
}

/**
 * Splits the particle budget across assignments, best-first.
 *
 * Every assignment is a separate optimisation problem with its own global
 * best - particles on different assignments must not attract each other, or
 * the swarm tears itself between incompatible structures. So this is N
 * independent sub-swarms sharing one dispatch.
 *
 * `weights` should be the free-parameter count. An equal split quietly favours
 * small models: an assignment with no free parameters is solved by its first
 * particle, while an eleven-parameter one is still searching when the run ends
 * and reports a correlation below what it could reach. The ranking then
 * measures how hard each assignment was to search rather than how well it fits.
 *
 * A wave holds only as many assignments as leaves room to weight. Packing in
 * `numParticles / minPer` of them spends the entire budget on the floor - with
 * 512 particles, a floor of 24 and 21 assignments, eight particles remain to
 * distribute and the weighting does nothing. Halving that keeps a real surplus,
 * at the cost of more waves.
 */
function allocateParticles(nAssign, numParticles, minPer = 24, weights = null) {
    const headroom = weights ? 2 : 1;
    const perWave = Math.max(1, Math.floor(numParticles / (minPer * headroom)));
    const waves = [];
    for (let i = 0; i < nAssign; i += perWave) {
        const ids = [];
        for (let k = i; k < Math.min(i + perWave, nAssign); k++) ids.push(k);
        waves.push({ ids, assignOf: weights
            ? weightedAssign(ids, numParticles, minPer, weights)
            : Uint32Array.from({ length: numParticles }, (_, p) => ids[p % ids.length]) });
    }
    return waves;
}

/**
 * Drops the worst assignments and re-runs the survivors with more particles.
 *
 * Successive halving: most assignments are visibly hopeless within a few tens
 * of generations, and the search is far better spent on the few that are not.
 * `best` is the best fitness seen per assignment so far.
 */
function pruneAssignments(ids, best, keepFraction = 0.5, minKeep = 4) {
    const ranked = [...ids].sort((a, b) => (best[b] ?? -Infinity) - (best[a] ?? -Infinity));
    const keep = Math.max(minKeep, Math.ceil(ranked.length * keepFraction));
    return ranked.slice(0, Math.min(keep, ranked.length));
}

/* ------------------------------------------------------------------ */

/**
 * "PbSO4" + Z -> [{ element, count }]. Handles nested groups and counts.
 */
function parseFormula(formula, Z = 1) {
    const counts = {};
    const parse = (s, mult) => {
        let i = 0;
        while (i < s.length) {
            if (s[i] === '(') {
                let depth = 1, j = i + 1;
                while (j < s.length && depth) { if (s[j] === '(') depth++; if (s[j] === ')') depth--; j++; }
                const inner = s.slice(i + 1, j - 1);
                const m = /^(\d+(?:\.\d+)?)/.exec(s.slice(j));
                const n = m ? parseFloat(m[1]) : 1;
                parse(inner, mult * n);
                i = j + (m ? m[1].length : 0);
                continue;
            }
            const m = /^([A-Z][a-z]?)(\d+(?:\.\d+)?)?/.exec(s.slice(i));
            if (!m) { i++; continue; }
            const el = m[1].toUpperCase();
            counts[el] = (counts[el] || 0) + mult * (m[2] ? parseFloat(m[2]) : 1);
            i += m[0].length;
        }
    };
    parse(formula.replace(/\s+/g, ''), Z);
    return Object.entries(counts).map(([element, count]) => ({ element, count }));
}

/**
 * Plausible Z values from the cell volume.
 *
 * A RANGE, not a number. The familiar 18 A^3 per non-hydrogen atom is an
 * organic-chemistry rule; dense inorganics run far tighter - PbSO4 sits near
 * 13 A^3 per atom, so the 18 rule alone returns Z=3 for a structure whose Z is
 * 4. Spanning 11 to 22 A^3 covers oxides, sulfates and intermetallics without
 * pretending to a precision the estimate does not have.
 *
 * Returns { min, max, likely } with `likely` the midpoint, and the caller is
 * expected to offer all of them: enumerating assignments for two or three Z is
 * cheap, and a wrong Z is otherwise a silent, total failure.
 */
function suggestZ(cellVolume, atomsPerFormula, symOpCount) {
    if (!cellVolume || !atomsPerFormula) return null;
    const lo = Math.max(1, Math.floor(cellVolume / (22 * atomsPerFormula)));
    const hi = Math.max(1, Math.ceil(cellVolume / (11 * atomsPerFormula)));
    const all = [];
    for (let z = lo; z <= hi; z++) {
        // Z is very often the group order or a divisor of it, because the
        // asymmetric unit usually holds a whole number of formula units.
        const divides = !symOpCount || symOpCount % z === 0 || z % symOpCount === 0;
        all.push({ Z: z, favoured: divides });
    }
    const fav = all.filter(a => a.favoured).map(a => a.Z);
    return { min: lo, max: hi, candidates: all.map(a => a.Z), favoured: fav,
             likely: fav.length ? fav[Math.floor(fav.length / 2)]
                                : Math.round((lo + hi) / 2) };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SHARKO_WYCKOFF_ASSIGN_VERSION,
        wyckoffFreedom, wyckoffProjector, projectOnto, wyckoffShift,
        enumerateAssignments, rankAssignments, buildAssignmentTables,
        projectSites, allocateParticles, weightedAssign, pruneAssignments,
        buildRestraintTables, packSymOps,
        parseFormula, suggestZ
    };
}
