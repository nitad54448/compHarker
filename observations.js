// Bumped whenever this file changes. Harko.html compares it against what it
// expects and says so if a browser cache is serving something older - a stale
// module reports errors at line numbers that no longer exist, which sends you
// looking for a bug that was already fixed.
const SHARKO_OBSERVATIONS_VERSION = '2026-08-09j';

/* ------------------------------------------------------------------
   Observation preparation for the Wyckoff structure search.

   Everything the correlation fitness needs reduces to two numbers per
   reflection: |Fo|^2 and the powder multiplicity m. How those are obtained
   depends on what the file happened to carry, and the routes are NOT
   interchangeable - applying a correction twice is as wrong as not applying
   it, and both look like a plausible number.

   The route that matters most here is the last one. A file with nothing but
   h k l, d and an intensity is fully usable: m follows from the Laue group,
   which is always known because the space group is, and Lp follows from d and
   the wavelength, which the user supplies in the UI. Measured against a file
   that states both columns explicitly, the reconstruction recovers m exactly
   for all 282 reflections and Lp to 0.067% - the residual being the refined
   zero shift, which no reconstruction can know about. The resulting fitness
   agreed to four decimal places.

   So the only genuinely unrecoverable case is a missing wavelength, which is
   worth a prompt rather than a silent degradation into a map dominated by its
   low-angle reflections.
   ------------------------------------------------------------------ */

/**
 * Powder multiplicity: the orbit size of a reflection under the Laue group.
 *
 * Friedel's law is included, so (h,k,l) and (-h,-k,-l) count once. Pass the
 * ROTATION parts of the space-group operators; the translations are irrelevant
 * to which reflections are equivalent in magnitude.
 */
function powderMultiplicity(h, k, l, rotations) {
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

/** 2-theta in degrees from a d-spacing, or null beyond the Ewald limit. */
function twoThetaFromD(d, lambda) {
    if (!(d > 0) || !(lambda > 0)) return null;
    const st = lambda / (2 * d);
    if (st >= 1) return null;             // d < lambda/2 cannot diffract
    return 2 * Math.asin(st) * 180 / Math.PI;
}

/**
 * Reduces a reflection list to |Fo|^2 and m by whichever route the data allows.
 *
 * @param reflections  parsed rows: {h,k,l} plus any of d, twoTheta, multiplicity,
 *                     lp, Ihkl, Fo, sigma
 * @param opts.rotations   rotation parts of the space-group operators (required
 *                         whenever the file does not state m)
 * @param opts.wavelength  Angstrom, from the UI. Required when the file states
 *                         neither |Fo| nor 2theta.
 * @param opts.polarisationK  1 for an unpolarised lab tube, 0 for neutron
 * @param opts.radiation   'xray' | 'neutron'
 *
 * @returns {{rows, route, warnings, errors}} rows carry Fo2, mult, d, twoTheta
 */
function normaliseObservations(reflections, opts = {}) {
    const rotations = opts.rotations || null;
    const lambda = Number(opts.wavelength);
    const radiation = opts.radiation || 'xray';
    // A neutron beam is unpolarised and there is no polarisation factor at all;
    // forcing K = 0 gives P = 1 rather than the 0.5-to-1 ramp an X-ray tube has.
    const K = radiation === 'neutron' ? 0
            : (Number.isFinite(opts.polarisationK) ? opts.polarisationK : 1);

    const warnings = [], errors = [];
    const src = reflections.filter(r => Number.isFinite(r.h) && Number.isFinite(r.k) &&
                                        Number.isFinite(r.l) && (r.h || r.k || r.l));
    if (!src.length) { errors.push('No usable reflections.'); return { rows: [], route: null, warnings, errors }; }

    const has = f => src.every(r => Number.isFinite(r[f]));
    const haveFo   = has('Fo');
    const haveM    = src.every(r => Number.isFinite(r.multiplicity) && r.multiplicity > 0);
    const haveLp   = src.every(r => Number.isFinite(r.lp) && r.lp > 0);
    const haveI    = has('Ihkl');
    const haveTth  = has('twoTheta');
    const haveD    = src.every(r => Number.isFinite(r.d) && r.d > 0);

    // d and 2theta are interconvertible given the wavelength, so fill whichever
    // is missing before choosing a route. This is what makes a file carrying
    // only d as good as one carrying 2theta.
    let derivedTth = 0, derivedD = 0;
    if (Number.isFinite(lambda) && lambda > 0) {
        for (const r of src) {
            if (!Number.isFinite(r.twoTheta) && Number.isFinite(r.d)) {
                const t = twoThetaFromD(r.d, lambda);
                if (t !== null) { r._tth = t; derivedTth++; }
            } else if (Number.isFinite(r.twoTheta) && !Number.isFinite(r.d)) {
                const th = 0.5 * r.twoTheta * Math.PI / 180;
                const s = Math.sin(th);
                if (s > 1e-9) { r._d = lambda / (2 * s); derivedD++; }
            }
        }
    }
    const tthOf = r => Number.isFinite(r.twoTheta) ? r.twoTheta : r._tth;
    const dOf   = r => Number.isFinite(r.d) ? r.d : r._d;
    const anyTth = src.every(r => Number.isFinite(tthOf(r)));

    // Multiplicity: the file's own value is authoritative, because it is what
    // the extraction assumed. Only compute it when the file is silent.
    let multSource = 'file';
    if (!haveM) {
        if (!rotations || !rotations.length) {
            errors.push('The file does not state the powder multiplicity and no symmetry ' +
                        'operators were supplied, so it cannot be reconstructed.');
            return { rows: [], route: null, warnings, errors };
        }
        multSource = 'computed';
        for (const r of src) r._mult = powderMultiplicity(r.h, r.k, r.l, rotations);
    }
    const multOf = r => haveM ? r.multiplicity : r._mult;

    let route, lpSource = 'n/a';
    const rows = [];

    if (haveFo) {
        // Powder 5 removed m and Lp itself, at full precision from the refined
        // 2theta including the zero shift. Nothing further is divided out.
        route = 'Fo';
        for (const r of src) rows.push({ h: r.h, k: r.k, l: r.l, Fo2: r.Fo * r.Fo,
                                         mult: multOf(r), d: dOf(r), twoTheta: tthOf(r),
                                         sigma: r.sigma });
    } else if (haveM && haveLp && haveI) {
        route = 'm_lp';
        for (const r of src) rows.push({ h: r.h, k: r.k, l: r.l,
                                         Fo2: r.Ihkl / (r.multiplicity * r.lp),
                                         mult: r.multiplicity, d: dOf(r), twoTheta: tthOf(r),
                                         sigma: r.sigma });
    } else if (haveI && anyTth) {
        // The reconstruction route. Lp from the stated model, m from the Laue
        // group. Verified against a file carrying both columns: m exact,
        // Lp to 0.067%.
        route = (haveTth ? 'lp_from_2theta' : 'lp_from_d');
        lpSource = 'computed';
        if (!haveTth && !(lambda > 0)) {
            errors.push('This file gives d but not 2theta, so the wavelength is needed ' +
                        'to remove the Lorentz-polarisation factor. Enter it in the UI.');
            return { rows: [], route: null, warnings, errors };
        }
        for (const r of src) {
            const tth = tthOf(r);
            const lp = sharkoLorentzPolarization(tth, K);
            rows.push({ h: r.h, k: r.k, l: r.l, Fo2: r.Ihkl / (multOf(r) * lp),
                        mult: multOf(r), d: dOf(r), twoTheta: tth, sigma: r.sigma });
        }
    } else if (haveI) {
        route = 'raw';
        for (const r of src) rows.push({ h: r.h, k: r.k, l: r.l, Fo2: r.Ihkl,
                                         mult: multOf(r), d: dOf(r), twoTheta: tthOf(r),
                                         sigma: r.sigma });
        warnings.push('Neither |Fo|, 2theta nor a wavelength is available, so the ' +
                      'Lorentz-polarisation factor could not be removed. The fitness will be ' +
                      'dominated by the low-angle reflections and the search may converge on a ' +
                      'structure that fits only those.');
    } else {
        errors.push('The reflection list carries no intensity column.');
        return { rows: [], route: null, warnings, errors };
    }

    const bad = rows.filter(r => !Number.isFinite(r.Fo2) || !(r.d > 0));
    if (bad.length) {
        warnings.push(`${bad.length} reflection(s) dropped: no finite intensity or d-spacing.`);
    }
    const clean = rows.filter(r => Number.isFinite(r.Fo2) && r.d > 0);
    if (!clean.length) { errors.push('No reflections survived normalisation.'); }

    if (derivedTth) warnings.push(`2theta derived from d for ${derivedTth} reflection(s) at lambda = ${lambda} A.`);
    if (multSource === 'computed') warnings.push('Powder multiplicities computed from the Laue group.');

    return { rows: clean, route, multSource, lpSource, warnings, errors,
             dMin: Math.min(...clean.map(r => r.d)), dMax: Math.max(...clean.map(r => r.d)) };
}

/**
 * Overall temperature factor from a Wilson plot.
 *
 * B is worth 0.31 in correlation on real PbSO4 data - 0.65 at B = 0 against
 * 0.97 at the optimum - so it is not optional. But it is also structure
 * independent: <|F|^2> over a shell depends only on the cell contents, so B
 * can be fixed once from the data and the composition before any search runs,
 * and never touched again.
 *
 *     ln( <Fo^2> / sum_j f_j(s)^2 )  =  ln K  -  2 B s^2/4
 *
 * The slope against (s/2)^2 gives -2B. Shells with too few reflections or a
 * non-positive mean are skipped rather than allowed to drag the fit.
 *
 * @param rows      from normaliseObservations()
 * @param demand    [{element, z, count}] - the cell contents
 * @param formFactor  (element, s) -> f, from scatterers.js. Falls back to Z.
 */
function wilsonB(rows, demand, formFactor, options = {}) {
    const nShells = options.shells ?? 10;
    const minPer = options.minPerShell ?? 5;

    const s = rows.map(r => 1 / r.d);
    const smin = Math.min(...s), smax = Math.max(...s);
    const span = Math.max(1e-9, smax - smin);

    const sumI = new Float64Array(nShells), sumS = new Float64Array(nShells);
    const cnt = new Int32Array(nShells);
    for (let i = 0; i < rows.length; i++) {
        const b = Math.min(nShells - 1, Math.floor((s[i] - smin) / span * nShells));
        sumI[b] += rows[i].Fo2; sumS[b] += s[i]; cnt[b]++;
    }

    const xs = [], ys = [];
    for (let b = 0; b < nShells; b++) {
        if (cnt[b] < minPer) continue;
        const sBar = sumS[b] / cnt[b];
        const meanI = sumI[b] / cnt[b];
        if (!(meanI > 0)) continue;
        let sigF2 = 0;
        for (const dmd of demand) {
            let f = formFactor ? formFactor(dmd.element, sBar) : null;
            if (!Number.isFinite(f)) f = dmd.z;
            sigF2 += dmd.count * f * f;
        }
        if (!(sigF2 > 0)) continue;
        const stol2 = (sBar / 2) * (sBar / 2);
        xs.push(stol2);
        ys.push(Math.log(meanI / sigF2));
    }

    if (xs.length < 3) {
        return { B: 0, scale: 1, ok: false,
                 note: `Only ${xs.length} usable shell(s); B left at 0. ` +
                       `A wrong B lowers the peak correlation but barely moves where it is, ` +
                       `so this degrades the search rather than breaking it.` };
    }

    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < xs.length; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
    const n = xs.length;
    const denom = n * sxx - sx * sx;
    if (Math.abs(denom) < 1e-12) return { B: 0, scale: 1, ok: false, note: 'Degenerate Wilson fit.' };
    const slope = (n * sxy - sx * sy) / denom;
    const intercept = (sy - slope * sx) / n;

    // slope = -2B. A negative B would mean intensity RISING with angle, which
    // is unphysical for a static structure and usually means the data has been
    // sharpened already or the corrections were applied twice. Clamp and say so.
    let B = -slope / 2;
    let note = null;
    if (!(B > 0)) {
        note = `The Wilson slope gives B = ${B.toFixed(2)}, which is unphysical. ` +
               `Intensity rising with angle usually means Lp or a sharpening was applied twice. ` +
               `B has been clamped to 0.`;
        B = 0;
    } else if (B > 20) {
        note = `B = ${B.toFixed(1)} is very high; check that Lp was removed.`;
    }
    return { B, scale: Math.exp(intercept), ok: true, shells: n, note };
}

/* ------------------------------------------------------------------
   Powder residuals and model comparison.

   The Patterson correlation the search maximises cannot settle which of
   several assignments is right. It is computed on a map dominated by the
   heavy-atom vectors, so two models that agree on the lead and differ on the
   oxygens score almost identically - on real PbSO4 the correct assignment
   came fourth, behind two models scoring 0.9872 to its 0.9804. Worse, the
   correlation says nothing about the PRICE of a fit: a model free to move
   more atoms will fit at least as well as one that cannot, so comparing raw
   agreement across assignments of different dimension is not a comparison at
   all.

   R on |F| answers the first problem, because every reflection counts rather
   than every grid point. Hamilton's ratio test answers the second, because it
   asks how much better a fit has to be before the extra freedom that bought it
   is justified.
   ------------------------------------------------------------------ */

/**
 * Groups reflections into the powder lines actually measured.
 *
 * Reflections closer in d than the overlap tolerance are one observation, not
 * several: their intensities were never separately measured, and counting them
 * separately inflates the observation count that every significance test
 * divides by. On the PbSO4 data 282 reflections are 182 lines.
 *
 * MUST match swPackReflections() in swarm_wyckoff.js, which is why the
 * tolerance is passed in from the solver's own result rather than repeated
 * here as a literal.
 *
 * @returns [[rowIndex, ...], ...] in descending d
 */
function groupPowderObservations(rows, overlapTol = 0.002) {
    const order = rows.map((r, i) => i).sort((a, b) => rows[b].d - rows[a].d);
    const groups = [];
    let cur = null, curD = 0;
    for (const i of order) {
        if (cur && Math.abs(rows[i].d - curD) / curD < overlapTol) {
            cur.push(i);
        } else {
            cur = [i]; curD = rows[i].d; groups.push(cur);
        }
    }
    return groups;
}

/**
 * Powder R factors, formed on the LINES rather than the reflections.
 *
 * An overlapped group contributes one measured intensity constraining several
 * |F|, so the residual is taken on the group sums - the same quantity the
 * search's fitness uses. Comparing individual |Fo| inside an overlap would be
 * comparing numbers that came out of a decomposition, not out of the
 * diffractometer.
 *
 * Both are returned because they answer different questions. R is the number
 * a crystallographer reads. wR is the one Hamilton's test is defined on: the
 * test is derived from a weighted least-squares residual, and applying it to
 * an unweighted R is not the same statistic.
 *
 * The weights here are unity on the group amplitudes. That is a real
 * assumption - Hamilton's distribution theory is exact only when the weights
 * are the inverse covariance of the observations - so the p-values below are
 * indicative rather than authoritative. Unit weights are still the honest
 * default when the file carries no sigmas, which is the usual case.
 *
 * @param Fc     |Fc| per row, from structureFactors()
 * @param rows   from normaliseObservations(); mult is the powder multiplicity
 * @param groups from groupPowderObservations()
 */
function powderResiduals(Fc, rows, groups) {
    const n = groups.length;
    const Ao = new Float64Array(n), Ac = new Float64Array(n);
    for (let g = 0; g < n; g++) {
        let io = 0, ic = 0;
        for (const i of groups[g]) {
            const m = rows[i].mult || 1;
            io += m * Math.max(0, rows[i].Fo2);
            ic += m * Fc[i] * Fc[i];
        }
        Ao[g] = Math.sqrt(io); Ac[g] = Math.sqrt(ic);
    }
    let sAoAc = 0, sAc2 = 0;
    for (let g = 0; g < n; g++) { sAoAc += Ao[g] * Ac[g]; sAc2 += Ac[g] * Ac[g]; }
    if (!(sAc2 > 0)) return null;
    const k = sAoAc / sAc2;          // one scale factor, refined by least squares

    let num = 0, den = 0, wnum = 0, wden = 0;
    for (let g = 0; g < n; g++) {
        const dlt = Ao[g] - k * Ac[g];
        num += Math.abs(dlt); den += Ao[g];
        wnum += dlt * dlt;  wden += Ao[g] * Ao[g];
    }
    if (!(den > 0) || !(wden > 0)) return null;
    return { R: num / den, wR: Math.sqrt(wnum / wden), scale: k, nObs: n };
}

/* ---- Regularized incomplete beta, for the F distribution ---------------
   Lentz's continued fraction, Numerical Recipes 6.4. Checked against
   tabulated F critical values: F(1,10), F(2,20), F(3,60), F(4,171), F(1,1000)
   and F(11,180) at alpha = 0.05 all return a tail of 0.050.
   ---------------------------------------------------------------------- */
function sharkoLogGamma(z) {
    const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
               -176.61502916214059, 12.507343278686905, -0.13857109526572012,
               9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - sharkoLogGamma(1 - z);
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < 8; i++) x += g[i] / (z + i + 1);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function sharkoBetaCF(a, b, x) {
    const TINY = 1e-300, EPS = 3e-16, MAXIT = 300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
        const m2 = 2 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
        c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
        c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < EPS) break;
    }
    return h;
}

function sharkoBetaI(a, b, x) {
    if (!(x > 0)) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(sharkoLogGamma(a + b) - sharkoLogGamma(a) - sharkoLogGamma(b) +
                        a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2)
        ? bt * sharkoBetaCF(a, b, x) / a
        : 1 - bt * sharkoBetaCF(b, a, 1 - x) / b;
}

/** P(F > f) for an F distribution with d1, d2 degrees of freedom. */
function sharkoFTail(f, d1, d2) {
    if (!(f > 0) || !(d1 > 0) || !(d2 > 0)) return 1;
    return sharkoBetaI(d2 / 2, d1 / 2, d2 / (d2 + d1 * f));
}

/**
 * Hamilton's R-ratio test.
 *
 * Two models fitted to the same data, one with more parameters than the other.
 * The larger model always fits at least as well; the question is whether it
 * fits enough better to be worth the freedom. Under the hypothesis that the
 * SMALLER model is correct,
 *
 *     (wR_small / wR_large)  ~  [ 1 + (b / nu) F(b, nu) ]^(1/2)
 *
 * with b the number of parameters given up and nu = n - m the degrees of
 * freedom of the larger model. Exceeding the critical ratio rejects the
 * smaller model - the extra parameters earned their place.
 *
 * Returns null where the test does not apply, which is not a failure: two
 * models of EQUAL dimension are not nested and Hamilton says nothing about
 * them, and a model that fits worse while spending more parameters needs no
 * test at all - it is dominated outright.
 *
 * @param wrSmall/pSmall  the more restrictive model
 * @param wrLarge/pLarge  the freer one
 * @param nObs            independent observations (powder LINES, not reflections)
 */
function hamiltonRatioTest(wrSmall, pSmall, wrLarge, pLarge, nObs) {
    const b = pLarge - pSmall;
    const nu = nObs - pLarge;
    if (!(b > 0) || !(nu > 0)) return null;
    if (!(wrSmall > 0) || !(wrLarge > 0)) return null;
    const ratio = wrSmall / wrLarge;
    // The observed F implied by the ratio, inverting the relation above.
    const f = (ratio * ratio - 1) * nu / b;
    return {
        b, nu, ratio,
        f: Math.max(0, f),
        // p is the probability of seeing a ratio this large if the smaller
        // model were the true one.
        p: sharkoFTail(f, b, nu),
        critical95: Math.sqrt(1 + (b / nu) * sharkoFInverse95(b, nu))
    };
}

/**
 * F value with a 5% upper tail, by bisection on sharkoFTail.
 *
 * Only used to report the critical ratio alongside the p-value, so a slow
 * exact inversion is preferable to a table that would have to be interpolated.
 */
function sharkoFInverse95(d1, d2) {
    let lo = 0, hi = 1000;
    for (let i = 0; i < 80; i++) {
        const mid = 0.5 * (lo + hi);
        if (sharkoFTail(mid, d1, d2) > 0.05) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SHARKO_OBSERVATIONS_VERSION, powderMultiplicity, twoThetaFromD, normaliseObservations,
                       wilsonB, groupPowderObservations, powderResiduals, hamiltonRatioTest,
                       sharkoFTail, sharkoBetaI };
}
