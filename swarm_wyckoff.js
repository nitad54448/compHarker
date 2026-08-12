// Bumped whenever this file changes. Harko.html compares it against what it
// expects and says so if a browser cache is serving something older - a stale
// module reports errors at line numbers that no longer exist, which sends you
// looking for a bug that was already fixed.
const SHARKO_SWARM_WYCKOFF_VERSION = '2026-08-09j';

/* ------------------------------------------------------------------
   swarm_wyckoff.js - the whole Wyckoff-constrained structure search.

   Harko.html calls runWyckoffSearch() once and gets ranked structures back.
   Everything between - normalising the observations, fixing the temperature
   factor, enumerating Wyckoff assignments, running the swarm and scoring the
   result - lives here rather than in the page.

   Requires, already loaded: symmetry_utils.js, wyckoff_assign.js,
   cc_fitness.js, observations.js, scatterers.js.

   THE SHAPE OF THE SEARCH

   Each particle carries an ASSIGNMENT - a choice of Wyckoff position for every
   independent site - so the cell contents are the requested composition by
   construction and cannot drift. Assignments are searched simultaneously in
   one dispatch, as independent sub-swarms: particles on different assignments
   must never attract each other, or the swarm tears itself between
   incompatible structures.

   Fitness runs in two phases. Phase 1 is the Patterson vector sum already in
   the retired swarm_multi.wgsl, which is smooth and gets particles into the right
   neighbourhood cheaply. Phase 2 is the multiplicity-weighted correlation of
   |F|^2, which IS the Patterson map correlation by Parseval and is what
   actually decides between near-solutions. Measured on real PbSO4 data the
   correct structure scores 0.96 against 0.30 for random ones - a 5.9 sigma
   separation over 200 trials.

   Assignments are raced rather than each given a full budget: most are
   visibly hopeless within a few tens of generations.
   ------------------------------------------------------------------ */

const SW_WG = 64;

/* ------------------------------------------------------------------
   Search tunables, in one place.

   These were scattered through the file as `?? 0.002`, `|| 512`, `?? 24` and
   bare locals inside the update loop, which made "what does this default to"
   a question you answered by reading the whole file. Every one is still
   overridable per run through the options object; this is only where the
   fallback lives.
   ------------------------------------------------------------------ */
const SW_DEFAULTS = Object.freeze({
    // --- Observations ---
    overlapTol: 0.002,      // fractional d spacing within which powder lines are one observation

    // --- Wyckoff enumeration ---
    maxSitesPerElement: 4,  // raised automatically when the composition demands more
    maxRepeatPerPosition: 4,

    // --- Swarm size ---
    numParticles: 512,
    generations: 400,
    minParticlesPerAssignment: 24,

    // --- Resolution ramp ---
    rampStart: 0.25,        // fraction of the groups active at generation 0
    rampFull: 0.6,          // fraction of the run by which all groups are active

    // --- Penalties, in CC units ---
    penClash: 0.05,         // per clashing pair
    // Per Angstrom of unmet distance constraint. Raised from 0.02, which was
    // set when the only upper-bound rule was a loose "some O within 1.65 A"
    // nearest-neighbour test. As the restoring force of a coordination
    // constraint it was far too weak to argue with the correlation: it held
    // PbSO4's sulfate at 2.00 A against a window ending at 1.90 for an entire
    // run. The ramp keeps it gentle early - 0.2 x this at generation 0 - so
    // exploration is not what pays for it.
    penBond: 0.10,
    penCoord: 0.03,         // per neighbour missing from, or surplus to, a coordination number
    // The penalty ramp. Everything above is multiplied by a factor sweeping
    // from penRampStart to penRampEnd over the run: soft early, so the swarm
    // has a gradient out of the clashes every random start produces; decisive
    // late, so no impossible structure survives on correlation alone.
    penRampStart: 0.2,
    penRampEnd: 4.0,

    // --- Metropolis update ---
    // Replaced particle swarm. PSO's social term pulls every particle of an
    // assignment toward that assignment's best, so the population collapses
    // onto whichever basin was found first and the restarts exist to undo it.
    // On PbSO4 that showed as a run-to-run coin flip between structures with
    // Pb-O at 1.01 A and the true one - the correct answer scores BEST on both
    // CC and R when it is found, so the objective was never the problem and
    // the sampling was. Independent chains cannot collapse: each one keeps its
    // own state and its own step size, and nothing shares a direction.
    // Replica-exchange ladder, in correlation units. Temperatures are FIXED,
    // not annealed: a single cooling chain that anneals into a wrong basin has
    // no way back out, which is exactly what the PbSO4 logs showed - converged,
    // at the bottom of its basin, 0.016 in CC below a structure known to exist.
    // A ladder keeps hot replicas roaming for the whole run and swaps their
    // discoveries downward.
    tempHot: 0.05,          // top rung: moves freely, refines nothing
    tempCold: 5e-4,         // bottom rung: refines, barely explores
    rungs: 8,               // geometric between the two
    swapEvery: 10,          // generations between exchange sweeps
    stepInit: 0.08,         // proposal width, fractional coordinates
    stepMin: 2e-4,
    stepMax: 0.5,
    targetAccept: 0.3,      // Robbins-Monro target for the per-chain step size
    stepAdapt: 0.08,        // how hard the step size chases that target
    // The resolution ramp is held constant over a block of generations so a
    // proposal and the state it is tested against share one objective. More
    // steps means a smoother ramp and more re-measurements of the current
    // state: one extra dispatch each.
    rampSteps: 24,

    // --- Final quench ---
    // A greedy descent from each assignment's best, after the restarts. See
    // quench() for why the search cannot produce this itself.
    // Path length, not breadth. A greedy descent walks DOWN a valley one move
    // at a time, so 23 chains of 150 steps and 1 chain of 3450 are not the same
    // purchase: measured on the real PbSO4 data from a rank-1 structure at
    // R 11.9%, 23x150 reached 10.5%, 23x600 reached 10.0%, and 1x10000 reached
    // 9.9% on a third of the evaluations. Chains only help by trying different
    // directions from the same point; they cannot walk further for you.
    quenchSteps: 600,
    quenchStep0: 0.02,      // starting move size, fractional coordinates
    quenchStep1: 5e-5,      // and where it ends: well under the coordinate precision reported

    // --- Reporting ---
    topN: 20
});

/**
 * Largest symmetry-expanded atom count the device's workgroup memory allows.
 *
 * The CC kernel keeps five arrays per generated atom (x, y, z, type, and the
 * reduction scratch) rather than the six of the vector-sum kernel, so 20 bytes
 * per atom is the honest figure. The 0.85 factor leaves room for the compiler's
 * own workgroup allocations, which are not visible from here.
 */
function swMaxGenAtoms(device, floor = 128) {
    try {
        const BYTES_PER_ATOM = 20, REDUCTION_BYTES = SW_WG * 4 * 6;
        const budget = device?.limits?.maxComputeWorkgroupStorageSize || 16384;
        const cap = Math.floor((budget * 0.85 - REDUCTION_BYTES) / BYTES_PER_ATOM);
        return Math.max(floor, Math.min(4096, cap));
    } catch (e) {
        return floor;
    }
}

function swInject(src, maxGen, minImageShell = 0) {
    let out = src.replace(
        /const\s+MAX_GEN_ATOMS\s*:\s*u32\s*=\s*\d+u\s*;\s*\/\/__MAX_GEN_ATOMS__/,
        `const MAX_GEN_ATOMS: u32 = ${maxGen}u; //__injected__`);
    if (out === src) throw new Error('MAX_GEN_ATOMS marker not found in the kernel source.');

    const before = out;
    out = out.replace(
        /const\s+MIN_IMAGE_SHELL\s*:\s*i32\s*=\s*-?\d+\s*;\s*\/\/__MIN_IMAGE_SHELL__/,
        `const MIN_IMAGE_SHELL: i32 = ${minImageShell}; //__injected__`);
    if (out === before) throw new Error('MIN_IMAGE_SHELL marker not found in the kernel source.');
    return out;
}

/**
 * Is (-I | 0) one of the space group's operators?
 *
 * If it is, the cell contents the kernel generates are closed under inversion
 * through the origin: every atom at r has a partner at -r of the same element,
 * their sine terms cancel exactly, and F is real. The kernel then skips the
 * imaginary part outright - half the transcendentals in its hottest loop.
 *
 * TESTED, NOT ASSUMED, for two reasons. Being a centrosymmetric group is not
 * enough: the inversion centre has to be AT THE ORIGIN, and a setting that puts
 * it elsewhere carries (-I | t) with t nonzero, for which F is complex. And
 * powder data does not make this true by itself - Friedel's law makes the
 * PATTERN centrosymmetric, |F(h)| = |F(-h)|, which is a statement about
 * measured intensities and says nothing about whether F has an imaginary part.
 * A non-centrosymmetric structure measured on a powder still has one, and it
 * sits inside |F|^2 where it cannot be dropped.
 */
function hasInversionAtOrigin(symOps) {
    const nearInt = v => Math.abs(v - Math.round(v)) < 1e-6;
    return (symOps || []).some(op => {
        const r = op.r, t = op.t || [0, 0, 0];
        if (!r || r.length < 9) return false;
        for (let i = 0; i < 9; i++) {
            const want = (i % 4 === 0) ? -1 : 0;    // -I, row-major
            if (Math.abs(r[i] - want) > 1e-6) return false;
        }
        return nearInt(t[0]) && nearInt(t[1]) && nearInt(t[2]);
    });
}

/**
 * One standard normal deviate, Marsaglia polar.
 *
 * A Gaussian proposal rather than a uniform one because the acceptance test
 * assumes a symmetric kernel and a Gaussian keeps that property under the
 * projection onto a Wyckoff subspace, which a box does not - an axis-aligned
 * box projected onto a slanted subspace is no longer axis-aligned or uniform.
 *
 * WHY THIS FORM. This function is the single most expensive thing in the whole
 * search, which is not obvious and was measured rather than guessed: at 8192
 * chains the host-side half of a generation costs about 9 ms and 8.6 ms of that
 * is here. It is called once per coordinate per chain per step - 123,000 times
 * a generation at 8192 chains and five sites.
 *
 * Box-Muller was costing a log, a sqrt, a cos and two Math.random() calls per
 * deviate, and throwing away the sine half of the pair it computes. The polar
 * form has no trigonometry at all: it rejects about 21% of its samples for
 * landing outside the unit circle, and that is still cheaper than a cosine.
 * Both deviates are kept. Measured at 8192 chains: 8.6 ms -> 6.1 ms.
 *
 * The distribution is exactly normal, not an approximation. That matters: a
 * bounded proposal kernel would never make a long jump, and long jumps are how
 * a hot replica leaves a basin.
 */
let swSpareGaussian = 0, swHasSpare = false;
function swGaussian() {
    if (swHasSpare) { swHasSpare = false; return swSpareGaussian; }
    let u, v, q;
    do {
        u = 2 * Math.random() - 1;
        v = 2 * Math.random() - 1;
        q = u * u + v * v;
    } while (q === 0 || q >= 1);
    const f = Math.sqrt(-2 * Math.log(q) / q);
    swSpareGaussian = v * f;
    swHasSpare = true;
    return u * f;
}

// The two Robbins-Monro multipliers, one for an accepted proposal and one for
// a rejected one. See the adaptation step in the search loop.
const SW_STEP_UP = Math.exp(SW_DEFAULTS.stepAdapt * (1 - SW_DEFAULTS.targetAccept));
const SW_STEP_DOWN = Math.exp(SW_DEFAULTS.stepAdapt * (0 - SW_DEFAULTS.targetAccept));

function swBuffer(device, data, usage) {
    const buf = device.createBuffer({
        size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
        usage: usage | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

/* ------------------------------------------------------------------ */
/*  Grouping and packing                                               */
/* ------------------------------------------------------------------ */

/**
 * Groups reflections a powder pattern cannot resolve, and packs for the kernel.
 *
 * Pawley's split of intensity between overlapped lines is a fitting artefact,
 * not a measurement, so lines closer in d than the overlap tolerance are summed
 * and compared as one observation. Groups come out sorted by d* ascending,
 * which makes the resolution ramp a single bound on the group index.
 *
 * The observation is the multiplicity-WEIGHTED |Fo|^2. That weighting is not a
 * choice: Parseval sums over the full sphere, where each unique reflection
 * appears m times, so m-weighting is exactly what makes this the map
 * correlation. Weighting by m*Lp instead - comparing raw peak areas - looks
 * better (0.99 against 0.96) but discriminates worse (3.2 sigma against 5.9),
 * because Lp spans a factor of 47 across the pattern and a handful of strong
 * low-angle lines come to dominate both the true and the random structures.
 */
function swPackReflections(rows, options = {}) {
    const tol = options.overlapTol ?? SW_DEFAULTS.overlapTol;
    const sorted = [...rows].sort((a, b) => b.d - a.d);   // d descending = d* ascending

    const groups = [];
    let cur = null;
    for (const r of sorted) {
        if (cur && Math.abs(r.d - cur.d) / cur.d < tol) {
            cur.members.push(r);
        } else {
            cur = { members: [r], d: r.d };
            groups.push(cur);
        }
    }

    const nGroups = groups.length;
    const nRefl = sorted.length;
    const reflPack = new Uint32Array(nRefl * 2);
    const groupMeta = new Float32Array(nGroups * 3);      // start, count, Iobs
    const groupD = new Float32Array(nGroups);
    const problems = [];

    let w = 0;
    groups.forEach((g, gi) => {
        let io = 0;
        groupMeta[gi * 3] = w;
        groupMeta[gi * 3 + 1] = g.members.length;
        for (const r of g.members) {
            if (Math.abs(r.h) > 511 || Math.abs(r.k) > 511 || Math.abs(r.l) > 511) {
                problems.push(`Reflection ${r.h} ${r.k} ${r.l} exceeds the +/-511 packing range.`);
                continue;
            }
            reflPack[w * 2] = ((r.h + 512) & 0x3FF) | (((r.k + 512) & 0x3FF) << 10)
                            | (((r.l + 512) & 0x3FF) << 20);
            reflPack[w * 2 + 1] = r.mult;
            io += r.mult * r.Fo2;
            w++;
        }
        groupMeta[gi * 3 + 2] = io;
        groupD[gi] = 1 / g.d;
    });

    return { reflPack, groupMeta, groupD, nRefl: w, nGroups, overlapTol: tol,
             overlapped: groups.filter(g => g.members.length > 1).length, problems };
}

/**
 * Scattering factors per (reflection, element type), folded with exp(-B s^2/4).
 *
 * On the MODEL side, never as a rescaling of the observations. Normalising the
 * data shell by shell to match point atoms was an earlier version of this and
 * it is wrong in a way that is easy to miss: it puts the true structure's
 * correlation at 0.88 instead of 1.000, because the data has been distorted and
 * the model has not.
 */
function swScatteringTable(refl, demand, options = {}) {
    const B = options.overallB ?? 0;
    const ff = options.formFactor || (() => null);
    const nElem = demand.length;
    const table = new Float32Array(refl.nRefl * nElem);

    const sOf = new Float32Array(refl.nRefl);
    for (let g = 0; g < refl.nGroups; g++) {
        const st = refl.groupMeta[g * 3], c = refl.groupMeta[g * 3 + 1];
        for (let m = 0; m < c; m++) sOf[st + m] = refl.groupD[g];
    }

    const missing = new Set();
    for (let r = 0; r < refl.nRefl; r++) {
        const s = sOf[r];
        const dw = B > 0 ? Math.exp(-B * s * s / 4) : 1;
        for (let e = 0; e < nElem; e++) {
            let f = ff(demand[e].element, s);
            if (!Number.isFinite(f)) { f = demand[e].z; missing.add(demand[e].element); }
            table[r * nElem + e] = f * dw;
        }
    }
    return { table, nElem, missing: [...missing] };
}

/**
 * How many reflection groups are active at a given point in the run.
 *
 * The |F|^2 landscape is far more oscillatory than the Patterson vector sum, so
 * a swarm turned loose on the full reflection set tends to stall in a local
 * maximum. Starting with the low-order data gives a smooth surface with few
 * maxima, and adding shells as the run proceeds sharpens it around whatever
 * basin the swarm has already found. Measured on real PbSO4 data at a 0.10
 * heavy-atom error, the full set reads 0.67 while the low-resolution quarter
 * reads 0.86 - the coarse surface still points uphill where the fine one has
 * begun to break up.
 *
 * Groups arrive sorted by d* ascending, so this is a single bound on the index.
 *
 * (cc_fitness.js has an identical copy. That module is the CPU reference
 * implementation used to validate the kernel; the app does not load it, and
 * duplicating thirty characters of arithmetic is better than making the page
 * fetch a second file for one function.)
 */
function rampedGroupCount(nGroups, generation, maxGen, startFrac = 0.25, fullBy = 0.6) {
    const t = Math.min(1, generation / Math.max(1, maxGen * fullBy));
    const frac = startFrac + (1 - startFrac) * t;
    return Math.max(8, Math.min(nGroups, Math.round(nGroups * frac)));
}

/* ------------------------------------------------------------------ */
/*  The search                                                         */
/* ------------------------------------------------------------------ */

/**
 * Runs the whole thing.
 *
 * @param {Object} o
 *   device            GPUDevice (already requested by the caller)
 *   ccShaderSource    text of swarm_cc.wgsl
 *   setting           one entry from sg/<n>.json settings[]
 *   cell              {a,b,c,alpha,beta,gamma}
 *   reflections       parsed rows, whatever columns the file had
 *   wavelength        Angstrom, from the UI
 *   formula, Z        e.g. 'PbSO4', 4
 *   atomData          {PB:{z,r}, S:{...}, O:{...}}
 *   windows           [{a,b,dmin,dmax,bothWays}]
 *   harkerSites       consolidated candidate positions, optional
 *   scatterTables     from loadScatteringTables(), optional
 *   radiation         'xray' | 'neutron'
 *   generations, numParticles, topN
 *   onProgress(info)  called each generation
 *   shouldStop()      polled each generation
 */
async function runWyckoffSearch(o) {
    const log = [];
    const say = m => { log.push(m); if (o.onLog) o.onLog(m); };

    /* ---- 1. Composition ---- */
    const comp = parseFormula(o.formula, o.Z);
    const demand = comp.map(c => {
        const ad = (o.atomData || {})[c.element];
        if (!ad) throw new Error(`No atomic data for element "${c.element}".`);
        if (!Number.isInteger(c.count)) {
            throw new Error(`${c.element} needs ${c.count} atoms per cell, which is not a whole ` +
                            `number. Partial occupancy is not supported by this search.`);
        }
        // `r` is the covalent radius (van der Waals only where none is listed).
        // It no longer sets the clash floor - that is the minimum-contact
        // slider, applied to every pair alike, see buildRestraintTables - and
        // is carried here only as per-element metadata for anything that wants
        // a size.
        return { element: c.element, count: c.count, z: ad.z, r: ad.rc ?? ad.r };
    });
    const nTot = demand.reduce((n, d) => n + d.count, 0);
    say(`${o.formula} with Z=${o.Z}: ${demand.map(d => d.element + d.count).join(' ')} ` +
        `= ${nTot} atoms per cell.`);

    /* ---- 2. Observations ---- */
    const rotations = (o.setting.sym_ops || []).map(op => op.r);
    const obs = normaliseObservations(o.reflections, {
        rotations, wavelength: o.wavelength, radiation: o.radiation,
        polarisationK: o.polarisationK
    });
    if (obs.errors.length) throw new Error(obs.errors.join(' '));
    obs.warnings.forEach(say);
    say(`${obs.rows.length} reflections, d ${obs.dMin.toFixed(2)}-${obs.dMax.toFixed(2)} A, ` +
        `route "${obs.route}".`);

    /* ---- 3. Temperature factor ---- */
    // Structure independent, so it is fixed once here rather than refined
    // during the search. On real PbSO4 data B is worth 0.31 in correlation
    // (0.65 at B=0 against 0.97 at the optimum), so leaving it at zero is not
    // an option; but the peak is broad, so a good estimate is enough.
  const ff = o.scatterTables
        ? makeFormFactor(o.scatterTables, { radiation: o.radiation || 'xray', ions: o.ions })
        : null;
    const wil = wilsonB(obs.rows, demand, ff);
    if (wil.note) say(wil.note);
    
    let overallB;
    if (Number.isFinite(o.overallB)) {
        overallB = o.overallB;
        say(`Overall B = ${overallB.toFixed(2)} (User defined). Wilson plot estimates ${wil.B.toFixed(2)}.`);
    } else {
        overallB = wil.B;
        say(`Overall B = ${overallB.toFixed(2)}${wil.ok ? ` (Wilson, ${wil.shells} shells)` : ''}.`);
    }
    /* ---- 4. Assignments ---- */
    if (!o.setting.wyckoff || !o.setting.wyckoff.length) {
        throw new Error(`Setting ${o.setting.symbol} carries no Wyckoff table. ` +
                        `Regenerate the database with cctbx_Harko_v1.py.`);
    }
    const en = enumerateAssignments(o.setting.wyckoff, demand, {
        maxSites: o.maxSitesPerElement ?? SW_DEFAULTS.maxSitesPerElement,
        maxRepeat: o.maxRepeat ?? SW_DEFAULTS.maxRepeatPerPosition,
        ceiling: o.wyckoffCapCeiling
    });
    if (en.error) throw new Error(en.error);
    let assignments = en.assignments;
    if (!assignments.length) throw new Error('No Wyckoff assignment matches this composition.');

    // Reduced basis, so every distance in this run - the pre-search ranking
    // here, and the kernel's contact tests below - uses the same, correct,
    // nearest-image convention.
    const red = sharkoReducedCell(o.cell);
    rankAssignments(assignments, {
        orth: red, harkerSites: o.harkerSites || [],
        heavyZ: Math.max(...demand.map(d => d.z)) * 0.7
    });
    say(`${assignments.length} Wyckoff assignment(s) consistent with the composition.`);

    /* ---- 5. Pack ---- */
    const T = buildAssignmentTables(assignments, (o.setting.sym_ops || []).length, demand);
    T.warnings.forEach(say);
    const refl = swPackReflections(obs.rows, { overlapTol: o.overlapTol });
    refl.problems.forEach(say);
    say(`${refl.nGroups} reflection group(s), ${refl.overlapped} containing overlaps.`);

    const ftab = swScatteringTable(refl, demand, { overallB, formFactor: ff });
    if (ftab.missing.length) {
        say(`No tabulated scattering factor for ${ftab.missing.join(', ')}; using f = Z. ` +
            `A missing element is indistinguishable from a wrong structure once it reaches ` +
            `the correlation, so check scatters/ is present.`);
    }
    // o.minContact is the Minimum contact distance slider. It was passed in
    // here and never read - it reached a params slot the kernel's struct does
    // not have, so it landed in padding. The slider moved and nothing changed,
    // while the post-search filter DID use it, which is how a run could spend
    // itself on contacts that were then all rejected.
    const restraints = buildRestraintTables(demand, o.windows || [],
                                            { minContact: o.minContact });
    restraints.problems.forEach(say);
    say(`Contact floor ${Number.isFinite(o.minContact) ? o.minContact.toFixed(2) : '1.00'} A on ` +
        `every pair${(o.windows || []).some(w => Number.isFinite(w.dmin))
            ? ', except pairs given their own dmin' : ''}.`);

    /* ---- 6. Device ---- */
    const device = o.device;
    if (!device) throw new Error('No GPUDevice supplied.');
    const maxGen = swMaxGenAtoms(device);
    if (nTot > maxGen) {
        throw new Error(`${nTot} atoms per cell exceeds this GPU's workgroup budget of ${maxGen}. ` +
                        `Reduce Z, or use a device with more workgroup storage.`);
    }

    // Minimum image. The kernel works in a reduced basis for the same lattice,
    // where rounding each fractional component independently is provably the
    // nearest image out to red.safeRadius. That covers every threshold for
    // almost any cell; for a genuinely thin one - a layered structure with a
    // short in-plane repeat - it does not, and the kernel is compiled to search
    // the 27 neighbouring translations instead, which is exact at any distance
    // and costs 27x in the contact loop only when it is actually needed.
    const needExact = restraints.maxDistance > red.safeRadius;
    const minImageShell = needExact ? 1 : 0;
    if (red.changed) {
        say(`Contact distances use a reduced lattice basis (safe to ` +
            `${red.safeRadius.toFixed(2)} A); in the cell's own basis the ` +
            `minimum-image convention is not reliable for a skewed cell.`);
    }
    if (needExact) {
        say(`This cell is thin: the reduced basis is only exact to ` +
            `${red.safeRadius.toFixed(2)} A and the restraints reach ` +
            `${restraints.maxDistance.toFixed(2)} A. The kernel will search the ` +
            `neighbouring lattice translations as well - correct, but slower.`);
    }

    const module = device.createShaderModule({
        code: swInject(o.ccShaderSource, maxGen, minImageShell)
    });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

    // Particle budget. Every assignment needs a floor of particles to behave as
    // a sub-swarm at all; that floor sets how many can run at once, and the
    // rest are queued into later waves in prior order.
    const numParticles = Math.max(SW_WG, o.numParticles || SW_DEFAULTS.numParticles);
    const minPer = o.minParticlesPerAssignment ?? SW_DEFAULTS.minParticlesPerAssignment;
    // Particles are shared out in proportion to the DIMENSIONALITY of each
    // assignment, not equally.
    //
    // An equal split systematically favours low-parameter models. A site on 4a
    // has no free parameter at all and a handful of particles finds its
    // optimum immediately; an assignment with eleven free parameters searching
    // the same budget is still wandering when the run ends, and reports a
    // correlation well below what it can actually reach. The ranking then
    // measures how hard each assignment was to search rather than how well it
    // fits - which is precisely backwards, since the extra parameters exist to
    // fit the data better.
    //
    // Weighting by parameter count is the cheapest correction that removes the
    // bias's direction. It does not remove it entirely - search difficulty
    // grows faster than linearly with dimension - so a close ranking between
    // assignments of very different size still deserves a longer run.
    const dims = assignments.map(A => Math.max(1, A.sites.reduce((n, s) => n + wyckoffFreedom(s.w), 0)));
    const waves = allocateParticles(assignments.length, numParticles, minPer, dims);
    const dimRange = `${Math.min(...dims)}-${Math.max(...dims)}`;
    say(`${numParticles} particles over ${waves.length} wave(s); assignments carry ` +
        `${dimRange} free parameters and receive particles in proportion.`);

    /* ---- 7. Static buffers ---- */
    const symPacked = packSymOps(o.setting.sym_ops);
    // groupData and the scattering table share one binding: WebGPU guarantees
    // only eight storage buffers per stage and this kernel uses all eight.
    const groupData = new Float32Array(refl.groupMeta.length + ftab.table.length);
    groupData.set(refl.groupMeta, 0);
    groupData.set(ftab.table, refl.groupMeta.length);
    const fTabOff = refl.groupMeta.length;

    const S = GPUBufferUsage.STORAGE;
    const bufGen   = swBuffer(device, T.genPack, S);
    const bufSym   = swBuffer(device, symPacked, S);
    const bufRefl  = swBuffer(device, refl.reflPack, S);


    const bufGroup = swBuffer(device, groupData, S);
    const bufTab   = swBuffer(device, restraints.tables, S);

    const coordsPerParticle = T.maxSites * 3;
    const positions = new Float32Array(numParticles * coordsPerParticle);
    const proposals = new Float32Array(numParticles * coordsPerParticle);
    const stepSize = new Float32Array(numParticles);
    const curFit = new Float32Array(numParticles);
    const curCC  = new Float32Array(numParticles).fill(NaN);
    const curPen = new Float32Array(numParticles);
    const tempOf = new Float32Array(numParticles);
    let acceptRate = NaN, swapRate = NaN;

    const bufPos       = swBuffer(device, positions, S | GPUBufferUsage.COPY_SRC);
    const bufAssign    = swBuffer(device, new Uint32Array(numParticles), S);
    const bufFit       = device.createBuffer({ size: numParticles * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    
    // New Bindings for GPU State
    const bufSiteProj  = swBuffer(device, T.siteProj, S);
    const bufStepSizes = swBuffer(device, stepSize, S);
const bufCurCC     = swBuffer(device, curCC, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const bufCurPen    = swBuffer(device, curPen, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const bufTempOf    = swBuffer(device, tempOf, S);
    const bufAccept    = swBuffer(device, new Uint32Array([0]), S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

    // Readback buffers for the Decoupled Batching
    const R = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    const bufRead      = device.createBuffer({ size: numParticles * 8, usage: R });
    const bufReadPos   = device.createBuffer({ size: numParticles * coordsPerParticle * 4, usage: R });
    const bufReadCC    = device.createBuffer({ size: numParticles * 4, usage: R });
    const bufReadPen   = device.createBuffer({ size: numParticles * 4, usage: R });
    const bufReadAccept= device.createBuffer({ size: 4, usage: R });

    const PARAM = Object.freeze({
        o0: 0, o1: 4, o2: 8,
        r0: 12, r1: 16, r2: 20,
        nTot: 24, maxSites: 25, numParticles: 26, nGroupsActive: 27,
        nElem: 28, nBondRules: 29, rMinOff: 30, ruleOff: 31,
        fTabOff: 32, nRefl: 33, penClash: 34, penBond: 35,
        penCoord: 36, penScale: 37, centro: 38, seed: 39, generation: 40, is_quench: 41
    });
    const params = new Float32Array(44);
    const paramsU32 = new Uint32Array(params.buffer);
    const bufParams = device.createBuffer({
        size: params.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    for (let i = 0; i < 3; i++) {
        const o4 = PARAM.o0 + i * 4, r4 = PARAM.r0 + i * 4;
        for (let k = 0; k < 3; k++) { params[o4 + k] = red.orth[i * 3 + k]; params[r4 + k] = red.xform[i * 3 + k]; }
    }
    params[PARAM.nTot] = nTot; params[PARAM.maxSites] = T.maxSites; params[PARAM.numParticles] = numParticles;
    params[PARAM.nElem] = restraints.nElem; params[PARAM.nBondRules] = restraints.nRules; params[PARAM.rMinOff] = restraints.rMinOff;
    params[PARAM.ruleOff] = restraints.ruleOff; params[PARAM.fTabOff] = fTabOff; params[PARAM.nRefl] = refl.nRefl;
    params[PARAM.penClash] = o.penClash ?? SW_DEFAULTS.penClash; params[PARAM.penBond] = o.penBond ?? SW_DEFAULTS.penBond;
    params[PARAM.penCoord] = o.penCoord ?? SW_DEFAULTS.penCoord; params[PARAM.penScale] = SW_DEFAULTS.penRampStart;
    params[PARAM.centro] = hasInversionAtOrigin(o.setting.sym_ops) ? 1 : 0;
    
    if (params[PARAM.centro]) say('Centrosymmetric group: F is real, so the kernel skips the imaginary part.');

    const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufPos } },
            { binding: 1, resource: { buffer: bufAssign } },
            { binding: 2, resource: { buffer: bufGen } },
            { binding: 3, resource: { buffer: bufSym } },
            { binding: 4, resource: { buffer: bufRefl } },
            { binding: 5, resource: { buffer: bufGroup } },
            { binding: 6, resource: { buffer: bufTab } },
            { binding: 7, resource: { buffer: bufFit } },
            { binding: 8, resource: { buffer: bufParams } },
            { binding: 9, resource: { buffer: bufSiteProj } },
            { binding: 10, resource: { buffer: bufStepSizes } },
            { binding: 11, resource: { buffer: bufCurCC } },
            { binding: 12, resource: { buffer: bufCurPen } },
            { binding: 13, resource: { buffer: bufTempOf } },
            { binding: 14, resource: { buffer: bufAccept } },
        ]
    });






    /* ---- 8. Run ---- */
    // Heavy-atom seeding lived here: it started a fraction of the particles
    // with the heaviest site on a consolidated Harker peak. Removed with the
    // checkbox that drove it. It existed to collapse the hardest part of a
    // GENERAL-POSITION search - finding the heavy atom's three free
    // coordinates - and the Wyckoff route does not have that problem, because
    // the assignment is what fixes the position. `harkerSites` is still
    // accepted and still used, by rankAssignments, to order the assignments
    // before the search; that use is unaffected.


    const generations = o.generations || SW_DEFAULTS.generations;
    const restarts = Math.max(1, o.restarts || 1);
    const results = [];
    let stopped = false;

    // Best over the WHOLE run, not the current wave. Reporting the wave's own
    // best made the fitness trace saw-tooth downwards: wave 1 holds the
    // most plausible assignments and scores highest, wave 4 holds the least
    // and starts from nothing, so the chart appeared to show the search
    // getting worse when it was simply starting a new problem.
    let runBestFit = -Infinity, runBestCC = NaN, runBestAssign = -1;

    for (let wi = 0; wi < waves.length && !stopped; wi++) {
        let ids = waves[wi].ids.slice();
        const assignOf = waves[wi].assignOf;

        // Per-assignment global bests. Particles on different assignments are
        // solving different problems, so a single shared best would pull each
        // sub-swarm toward a structure it cannot express.
        const gBestFit = new Float32Array(assignments.length).fill(-Infinity);
        const gBestCC  = new Float32Array(assignments.length).fill(NaN);
        const gBestPen = new Float32Array(assignments.length);   // raw, at weight 1
        const gBestPos = new Float32Array(assignments.length * coordsPerParticle);

        for (let i = 0; i < numParticles; i++) {
            const base = i * coordsPerParticle;
            for (let k = 0; k < coordsPerParticle; k++) positions[base + k] = Math.random();
            projectSites(positions, i, assignOf[i], T);
            curFit[i] = -Infinity; curCC[i] = NaN; curPen[i] = 0;
            stepSize[i] = SW_DEFAULTS.stepInit;
        }
        device.queue.writeBuffer(bufAssign, 0, assignOf);
        let ladders = [];

        /**
         * Builds the replica ladders and hands every chain its temperature.
         *
         * A ladder is `rungs` chains OF THE SAME ASSIGNMENT holding the
         * geometric temperature sequence from tempCold to tempHot. Chains of
         * different assignments are solving different problems in different
         * subspaces and their coordinates are not interchangeable, so an
         * exchange between them would be meaningless - the grouping by
         * assignment is what makes a swap legal.
         *
         * An assignment with a partial ladder left over puts those chains on the
         * COLD end. A partial ladder cannot exchange usefully, so the useful
         * thing for it to do is refine.
         */
        function buildLadders() {
            const byAssign = new Map();
            for (let i = 0; i < numParticles; i++) {
                const A = assignOf[i];
                if (!byAssign.has(A)) byAssign.set(A, []);
                byAssign.get(A).push(i);
            }
            const R = Math.max(2, SW_DEFAULTS.rungs);
            const ratio = Math.pow(SW_DEFAULTS.tempHot / SW_DEFAULTS.tempCold, 1 / (R - 1));
            const rungTemp = Array.from({ length: R },
                (_, r) => SW_DEFAULTS.tempCold * Math.pow(ratio, r));

            const out = [];
            for (const chains of byAssign.values()) {
                let c = 0;
                while (c + R <= chains.length) {
                    const lad = chains.slice(c, c + R);
                    lad.forEach((ci, r) => { tempOf[ci] = rungTemp[r]; });
                    out.push(lad);
                    c += R;
                }
                for (; c < chains.length; c++) tempOf[chains[c]] = rungTemp[0];
            }
            return out;
        }

        /**
         * One exchange sweep over every ladder.
         *
         * Adjacent rungs only, alternating which pairs are tried, which is the
         * standard way to keep the sweep a valid sequence of independent
         * two-body moves rather than a scramble.
         *
         *     P(swap) = min(1, exp[ (beta_cold - beta_hot) (f_hot - f_cold) ])
         *
         * So a hot replica that has found something better than the cold one
         * below it hands the structure down, and the cold replica's worse
         * structure goes up to be knocked about further. Nothing is lost and
         * nothing is duplicated: the two configurations trade places, the
         * temperatures stay with the rungs.
         */
        function exchangeSweep(parity, scale) {
            let tried = 0, taken = 0;
            const energy = i => Number.isFinite(curCC[i]) ? curCC[i] - curPen[i] * scale : curFit[i];
            for (const lad of ladders) {
                for (let r = parity; r + 1 < lad.length; r += 2) {
                    const a = lad[r], b = lad[r + 1];          // a colder than b
                    const fa = energy(a), fb = energy(b);
                    if (!Number.isFinite(fa) || !Number.isFinite(fb)) continue;
                    tried++;
                    const d = (1 / tempOf[a] - 1 / tempOf[b]) * (fb - fa);
                    if (d < 0 && Math.random() >= Math.exp(d)) continue;
                    taken++;
                    const ba = a * coordsPerParticle, bb = b * coordsPerParticle;
                    for (let k = 0; k < coordsPerParticle; k++) {
                        const t = positions[ba + k];
                        positions[ba + k] = positions[bb + k];
                        positions[bb + k] = t;
                    }
                    let t = curFit[a]; curFit[a] = curFit[b]; curFit[b] = t;
                    t = curCC[a];  curCC[a]  = curCC[b];  curCC[b]  = t;
                    t = curPen[a]; curPen[a] = curPen[b]; curPen[b] = t;
                    // The step size belongs to the RUNG, not to the structure:
                    // it is the move scale that rung's temperature accepts.
                }
            }
            if (tried) swapRate = taken / tried;
        }


        async function evaluateCoords(coords) {
            params[PARAM.is_quench] = 1.0;
            device.queue.writeBuffer(bufParams, 0, params);
            device.queue.writeBuffer(bufPos, 0, coords);
            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroups(numParticles);
            pass.end();
            enc.copyBufferToBuffer(bufFit, 0, bufRead, 0, numParticles * 8);
            device.queue.submit([enc.finish()]);
            await bufRead.mapAsync(GPUMapMode.READ);
            const out = new Float32Array(bufRead.getMappedRange().slice(0));
            bufRead.unmap();
            return out;
        }

        function recordBest(fitArr, coords, scale) {
            for (let i = 0; i < numParticles; i++) {
                const A = assignOf[i], base = i * coordsPerParticle;
                const f = fitArr[i];
                if (!Number.isFinite(f)) continue;
                const cc = fitArr[numParticles + i];
                const rawPen = Number.isFinite(cc) ? (cc - f) / scale : 0;
                const gRef = Number.isFinite(gBestCC[A]) ? gBestCC[A] - gBestPen[A] * scale
                                                        : gBestFit[A];
                if (f > gRef) {
                    gBestFit[A] = f; gBestCC[A] = cc; gBestPen[A] = rawPen;
                    gBestPos.set(coords.subarray(base, base + coordsPerParticle),
                                 A * coordsPerParticle);
                } else if (Number.isFinite(gRef)) {
                    gBestFit[A] = gRef;
                }
            }
        }

        async function quench(candidateIds) {
// ...
        }

        function checkSpread(fitArr) {
// ...
        }

      for (let restart = 0; restart < restarts && !stopped; restart++) {
        if (restart > 0) {
            for (let i = 0; i < numParticles; i++) {
                const base = i * coordsPerParticle;
                for (let k = 0; k < coordsPerParticle; k++) positions[base + k] = Math.random();
                projectSites(positions, i, assignOf[i], T);
                curFit[i] = -Infinity; curCC[i] = NaN; curPen[i] = 0;
                stepSize[i] = SW_DEFAULTS.stepInit;
            }
            say(`wave ${wi + 1}, restart ${restart + 1} of ${restarts}` +
                (Number.isFinite(acceptRate) ? `; acceptance ${(acceptRate * 100).toFixed(0)}%` : '') +
                (Number.isFinite(swapRate) ? `, exchange ${(swapRate * 100).toFixed(0)}%.` : '.'));
        }

        ladders = buildLadders();
        if (wi === 0 && restart === 0) {
            const R = Math.max(2, SW_DEFAULTS.rungs);
            say(`Replica exchange: ${ladders.length} ladder(s) of ${R} rungs, ` +
                `T ${SW_DEFAULTS.tempCold} to ${SW_DEFAULTS.tempHot}, ` +
                `swapping every ${SW_DEFAULTS.swapEvery} step(s).`);
        }

// Initialize chain state buffers for this restart
        device.queue.writeBuffer(bufTempOf, 0, tempOf);
        device.queue.writeBuffer(bufStepSizes, 0, stepSize);
        device.queue.writeBuffer(bufCurCC, 0, curCC);
        device.queue.writeBuffer(bufCurPen, 0, curPen);

        let firstEval = (wi === 0);
        let gensSinceSync = 0;

        for (let gen = 0; gen < generations; gen++) {
            if (o.shouldStop && o.shouldStop()) { stopped = true; break; }
            gensSinceSync++;

            const t = generations > 1 ? gen / (generations - 1) : 1;
            const rampLevel = Math.min(SW_DEFAULTS.rampSteps - 1, Math.floor(t * SW_DEFAULTS.rampSteps));
            params[PARAM.nGroupsActive] = rampedGroupCount(refl.nGroups, rampLevel, SW_DEFAULTS.rampSteps - 1, o.rampStart ?? SW_DEFAULTS.rampStart, o.rampFull ?? SW_DEFAULTS.rampFull);
            const scale = SW_DEFAULTS.penRampStart + t * (SW_DEFAULTS.penRampEnd - SW_DEFAULTS.penRampStart);
            
            // First evaluation check (to preserve checkSpread audit)
            if (firstEval) {
                const f0 = await evaluateCoords(positions);
                checkSpread(f0); firstEval = false;
                for (let i = 0; i < numParticles; i++) {
                    curCC[i] = f0[numParticles + i];
                    curPen[i] = (Number.isFinite(curCC[i]) && Number.isFinite(f0[i])) ? (curCC[i] - f0[i]) / scale : 0;
                    curFit[i] = f0[i];
                }
                device.queue.writeBuffer(bufCurCC, 0, curCC);
                device.queue.writeBuffer(bufCurPen, 0, curPen);
            }

            paramsU32[PARAM.seed] = Math.floor(Math.random() * 4294967296);
            paramsU32[PARAM.generation] = gen;
            params[PARAM.is_quench] = 0.0;
            params[PARAM.penScale] = scale;
            device.queue.writeBuffer(bufParams, 0, params);

            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroups(numParticles);
            pass.end();

            const isSyncGen = (ladders.length && (gen + 1) % SW_DEFAULTS.swapEvery === 0) || gen === generations - 1 || (o.onProgress && (gen % 10 === 0));
            

            if (isSyncGen) {
                enc.copyBufferToBuffer(bufFit, 0, bufRead, 0, numParticles * 8);
                enc.copyBufferToBuffer(bufPos, 0, bufReadPos, 0, numParticles * coordsPerParticle * 4);
                enc.copyBufferToBuffer(bufCurCC, 0, bufReadCC, 0, numParticles * 4);
                enc.copyBufferToBuffer(bufCurPen, 0, bufReadPen, 0, numParticles * 4);
                enc.copyBufferToBuffer(bufAccept, 0, bufReadAccept, 0, 4);
                device.queue.submit([enc.finish()]);

                await Promise.all([
                    bufRead.mapAsync(GPUMapMode.READ),
                    bufReadPos.mapAsync(GPUMapMode.READ),
                    bufReadCC.mapAsync(GPUMapMode.READ),
                    bufReadPen.mapAsync(GPUMapMode.READ),
                    bufReadAccept.mapAsync(GPUMapMode.READ)
                ]);

                const fp = new Float32Array(bufRead.getMappedRange().slice(0));
                positions.set(new Float32Array(bufReadPos.getMappedRange().slice(0)));
                curCC.set(new Float32Array(bufReadCC.getMappedRange().slice(0)));
                curPen.set(new Float32Array(bufReadPen.getMappedRange().slice(0)));
                const accepts = new Uint32Array(bufReadAccept.getMappedRange())[0];

                bufRead.unmap(); bufReadPos.unmap(); bufReadCC.unmap(); bufReadPen.unmap(); bufReadAccept.unmap();

                acceptRate = accepts / (numParticles * gensSinceSync);
                gensSinceSync = 0;
                device.queue.writeBuffer(bufAccept, 0, new Uint32Array([0]));

                for (let i = 0; i < numParticles; i++) {
                    curFit[i] = Number.isFinite(curCC[i]) ? curCC[i] - curPen[i] * scale : -Infinity;
                }
                
                // Track global best from the state just mapped back
                recordBest(fp, positions, scale);

                if (ladders.length && (gen + 1) % SW_DEFAULTS.swapEvery === 0) {
                    exchangeSweep(((gen + 1) / SW_DEFAULTS.swapEvery) % 2, scale);
                    device.queue.writeBuffer(bufPos, 0, positions);
                    device.queue.writeBuffer(bufCurCC, 0, curCC);
                    device.queue.writeBuffer(bufCurPen, 0, curPen);
                    // GPU completely owns StepSizes now. Do NOT overwrite them!
                }

                if (o.onProgress && (gen % 10 === 0 || gen === generations - 1)) {
                    let best = -Infinity, bestA = -1;
                    for (const a of ids) if (gBestFit[a] > best) { best = gBestFit[a]; bestA = a; }
                    if (best > runBestFit) {
                        runBestFit = best; runBestCC = gBestCC[bestA]; runBestAssign = bestA;
                    }

                    let bestSites = null;
                    if (bestA >= 0) {
                        const A = assignments[bestA];
                        const gb = bestA * coordsPerParticle;
                        bestSites = A.sites.map((st, si) => ({
                            element: st.element,
                            zn: st.z,
                            multiplicity: st.w.multiplicity,
                            wyckoff: `${st.w.multiplicity}${st.w.letter}`,
                            siteSymmetry: st.w.site_symmetry,
                            x: gBestPos[gb + si * 3],
                            y: gBestPos[gb + si * 3 + 1],
                            z: gBestPos[gb + si * 3 + 2]
                        }));
                    }
                    o.onProgress({ wave: wi + 1, waves: waves.length, generation: gen,
                                   generations, restart: restart + 1, restarts,
                                   acceptRate, swapRate,
                                   best: runBestFit, waveBest: best,
                                   cc: runBestCC, active: ids.length,
                                   assignment: bestA >= 0
                                       ? assignments[bestA].sites.map(x => `${x.element} ${x.w.multiplicity}${x.w.letter}`).join(', ')
                                       : null,
                                   bestSites });
                    await new Promise(r => setTimeout(r, 0));
                }
            } else {
                // Instantly submit generation to GPU without waiting for a readback!
                device.queue.submit([enc.finish()]);
            }

            // Successive halving.

            // Successive halving. Most assignments are visibly hopeless within
            // a few tens of generations, and the budget is far better spent on
            // the few that are not.
            //
            // Pruned once per WAVE, not once per restart. `ids` persists across
            // restarts, so a per-restart prune compounds: on PbSO4 it went
            // 35 -> 18 -> 9 -> 5 -> 4 over the first four restarts of ten, and
            // the field was decided before any restart had a chance to
            // converge. Worse, it is biased - an assignment with eleven free
            // parameters needs longer to look good than one with six, so the
            // flexible ones, which include the right answer for anything but
            // the simplest structure, are retired first.
            const pruneAt = Math.max(20, Math.round(generations * 0.45));
            if (restart === 0 && gen > 0 && gen === pruneAt && ids.length > 4) {
                const keep = pruneAssignments(ids, gBestFit, 0.5, 4);
                if (keep.length < ids.length) {
                    ids = keep;
                    // Reallocate by dimension, not round-robin. This is where the
                    // budget finally becomes generous - half the assignments have
                    // just retired - so distributing it uniformly here would undo
                    // the weighting at the moment it matters most.
                    const fresh = weightedAssign(ids, numParticles, minPer, dims);
                    for (let i = 0; i < numParticles; i++) {
                        const A = fresh[i];
                        if (assignOf[i] === A) continue;
                        assignOf[i] = A;
                        const base = i * coordsPerParticle;
                        // A reassigned chain starts fresh: its state belongs to a
                        // different subspace and is not a point of the new one.
                        for (let k = 0; k < coordsPerParticle; k++) {
                            positions[base + k] = Math.random();
                        }
                        projectSites(positions, i, A, T);
                        stepSize[i] = SW_DEFAULTS.stepInit;
                        curFit[i] = -Infinity; curCC[i] = NaN; curPen[i] = 0;
                    }
                    device.queue.writeBuffer(bufAssign, 0, assignOf);
                    // Those chains have no energy for their new subspace, so the
                    // next generation has to measure one before it can test
                    // anything against it.
                    needCurrentEval = true;
                    ladders = buildLadders();
                    say(`gen ${gen}: ${ids.length} assignment(s) still in contention.`);
                }
            }

            if (o.onProgress && (gen % 10 === 0 || gen === generations - 1)) {
                let best = -Infinity, bestA = -1;
                for (const a of ids) if (gBestFit[a] > best) { best = gBestFit[a]; bestA = a; }
                if (best > runBestFit) {
                    runBestFit = best; runBestCC = gBestCC[bestA]; runBestAssign = bestA;
                }

                // Hand out the running best as sites, so the caller can rebuild
                // the calculated and difference maps as the search proceeds.
                // Watching those two panes converge is how a wrong atom shows
                // itself - a correlation climbing on its own says only that
                // something is improving, not what.
                // zn is the atomic number; x, y, z are fractional coordinates.
                // Naming both "z" is how the two get silently swapped.
                let bestSites = null;
                if (bestA >= 0) {
                    const A = assignments[bestA];
                    const gb = bestA * coordsPerParticle;
                    bestSites = A.sites.map((st, si) => ({
                        element: st.element,
                        zn: st.z,
                        multiplicity: st.w.multiplicity,
                        wyckoff: `${st.w.multiplicity}${st.w.letter}`,
                        siteSymmetry: st.w.site_symmetry,
                        x: gBestPos[gb + si * 3],
                        y: gBestPos[gb + si * 3 + 1],
                        z: gBestPos[gb + si * 3 + 2]
                    }));
                }
                o.onProgress({ wave: wi + 1, waves: waves.length, generation: gen,
                               generations, restart: restart + 1, restarts,
                               // Whether the chains are moving at all. Without
                               // it a stuck run and a working one look the same
                               // from outside: both report a fitness that is not
                               // going up.
                               acceptRate, swapRate,
                               // `best` is the run's best so far, so the trace
                               // only ever climbs; `waveBest` is this wave's own.
                               best: runBestFit, waveBest: best,
                               cc: runBestCC, active: ids.length,
                               assignment: bestA >= 0
                                   ? assignments[bestA].sites.map(x => `${x.element} ${x.w.multiplicity}${x.w.letter}`).join(', ')
                                   : null,
                               bestSites });
                await new Promise(r => setTimeout(r, 0));
            }
        }

      }

        if (!stopped) await quench(waves[wi].ids);

        for (const A of waves[wi].ids) {
            if (!Number.isFinite(gBestFit[A]) || gBestFit[A] <= -Infinity) continue;
            // The reported penalty is the RAW one, at weight 1. Reporting it at
            // whatever the ramp happened to reach makes two candidates recorded
            // at different moments incomparable, and makes the number depend on
            // a tuning constant rather than on the structure.
            results.push({ assignIdx: A, score: gBestCC[A] - gBestPen[A],
                           cc: Number.isFinite(gBestCC[A]) ? gBestCC[A] : gBestFit[A],
                           penalty: Number.isFinite(gBestCC[A]) ? gBestPen[A] : 0,
                           coords: gBestPos.slice(A * coordsPerParticle,
                                                  (A + 1) * coordsPerParticle) });
        }
    }

    for (const b of [bufGen, bufSym, bufRefl, bufGroup, bufTab, bufPos, bufAssign,
                     bufFit, bufRead, bufParams]) { try { b.destroy(); } catch (e) {} }

    /* ---- 9. Report ---- */
    // Ranked by CC alone - the agreement between the observed and calculated
    // Patterson maps, which is the quantity that actually means something.
    //
    // The penalty weights are arbitrary: 0.05 per clash, 0.02 per Angstrom, no
    // more principled than any other pair of numbers. They earn their place
    // inside the search, where they keep particles away from nonsense and cost
    // nothing if slightly wrong, but ranking on them would let an arbitrary
    // constant decide which structure is reported. Distance windows are
    // enforced afterwards as a filter, which needs no weight at all.
    results.sort((a, b) => b.cc - a.cc);
    // More than the UI shows. Chemistry filtering happens in the caller, where
    // the contact code lives, and filtering a list of five can leave one.
    const topN = o.topN ?? SW_DEFAULTS.topN;
    const top = results.slice(0, topN).map(r => {
        const A = assignments[r.assignIdx];
        const sites = A.sites.map((s, si) => ({
            element: s.element,
            wyckoff: `${s.w.multiplicity}${s.w.letter}`,
            multiplicity: s.w.multiplicity,
            siteSymmetry: s.w.site_symmetry,
            // Occupancy is 1.0 and the multiplicity comes from the Wyckoff
            // letter. Scaling occupancy by mult/order - which a general-position
            // model needs - would double-count here.
            occupancy: 1.0,
            x: r.coords[si * 3], y: r.coords[si * 3 + 1], z: r.coords[si * 3 + 2]
        }));
        return { cc: r.cc, score: r.score, penalty: r.penalty, assignment: A.sites.map(s => `${s.element} ${s.w.multiplicity}${s.w.letter}`).join(', '),
                 harkerResidual: A.harkerResidual, sites, nFreeParams:
                     A.sites.reduce((n, s) => n + wyckoffFreedom(s.w), 0) };
    });

    say(top.length ? `Best score ${top[0].score.toFixed(4)} (CC ${top[0].cc.toFixed(4)}) for ${top[0].assignment}.`
                   : 'No candidate produced a finite score.');
    return { candidates: top, all: results.length, assignments: assignments.length,
             // How many the solver kept, so the caller can say so rather than
             // leaving the user to wonder why the table is shorter than the
             // number of assignments searched.
             kept: top.length, topN,
             // The clash floors the search actually enforced, so the caller's
             // filter can reject on the same numbers instead of its own.
             floors: restraints.floors,
             overallB, route: obs.route, reflections: obs.rows.length,
             groups: refl.nGroups, stopped, log,
             // The caller has to group the reflections the same way to count
             // independent observations for a significance test; handing back
             // the tolerance keeps the two from drifting apart.
             overlapTol: refl.overlapTol,
             // Kept so the caller can compute an R factor for whichever
             // candidate is selected, without re-normalising the file.
             obsRows: obs.rows, demand };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SHARKO_SWARM_WYCKOFF_VERSION, runWyckoffSearch, swPackReflections, swScatteringTable,
                       swMaxGenAtoms, rampedGroupCount };
}
