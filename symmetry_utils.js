/**
 * Shared helpers used by both sharko_worker.js (CPU path) and the main-thread
 * WebGPU Patterson map calculation. Kept in one file so the two code paths can
 * never drift apart on how a space-group setting is chosen, how reflections get
 * expanded, or how the cell metric is computed.
 *
 * Loaded by the worker via importScripts('symmetry_utils.js') and by the main
 * page via <script src="symmetry_utils.js"></script>.
 *
 * The space-group database is the JSON written by cctbx_generate_sg_harker_v6.py:
 *
 *   { schema_version: 7,
 *     space_groups: {
 *       "62": { number, standard_symbol, crystal_system, point_group,
 *               laue_class, centrosymmetric, chiral,
 *               settings: [ { symbol, description, hall, reflection_conditions,
 *                             harker_sections, rotations, sym_ops, order_z,
 *                             order_p, centering, centring_translations,
 *                             wyckoff } ] } } }
 *
 * `sym_ops` is the full operator list (rotation + translation, centring
 * operators included), `rotations` is the older rotation-only list. The
 * generator writes an empty `sym_ops` when the cctbx accessor is unavailable
 * in the build that produced the file, so a missing operator list is a real
 * possibility and must never pass silently.
 */

/* ------------------------------------------------------------------ */
/*  Cell geometry                                                      */
/* ------------------------------------------------------------------ */

/** Unit-cell volume in A^3 (general triclinic formula). */
function sharkoCellVolume(cell) {
    if (!cell) return NaN;
    const d2r = Math.PI / 180;
    const ca = Math.cos((cell.alpha ?? 90) * d2r);
    const cb = Math.cos((cell.beta  ?? 90) * d2r);
    const cg = Math.cos((cell.gamma ?? 90) * d2r);
    const disc = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg;
    if (!(disc > 0)) return NaN;
    return cell.a * cell.b * cell.c * Math.sqrt(disc);
}

/**
 * Fractional -> Cartesian matrix (row-major 3x3), standard setting with
 * a along x and b in the xy plane.
 */
function sharkoOrthMatrix(cell) {
    const d2r = Math.PI / 180;
    const al = (cell.alpha ?? 90) * d2r, be = (cell.beta ?? 90) * d2r, ga = (cell.gamma ?? 90) * d2r;
    const cosA = Math.cos(al), cosB = Math.cos(be), cosG = Math.cos(ga), sinG = Math.sin(ga);
    const V = sharkoCellVolume(cell);
    return new Float32Array([
        cell.a, cell.b * cosG, cell.c * cosB,
        0,      cell.b * sinG, cell.c * (cosA - cosB * cosG) / sinG,
        0,      0,             V / (cell.a * cell.b * sinG)
    ]);
}

/**
 * Cartesian length of a fractional difference vector, minimum-image applied.
 *
 * VALID ONLY FOR NEAR-ORTHOGONAL CELLS, or for separations below the safe
 * radius of sharkoReducedCell(). Rounding each fractional component
 * independently picks the representative inside the box [-1/2, 1/2)^3, which is
 * a fundamental domain but is NOT the Wigner-Seitz cell: in a skewed lattice a
 * neighbouring image can be closer than the one the box selects, and the
 * function then returns a distance that is too large.
 *
 * For anything that decides an outcome - a clash penalty, a bond restraint -
 * go through sharkoReducedCell() and use the reduced basis, where the box does
 * contain the ball of radius safeRadius and the two agree.
 */
function sharkoFracToCartLength(dx, dy, dz, orthMat) {
    dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
    const cx = orthMat[0] * dx + orthMat[1] * dy + orthMat[2] * dz;
    const cy = orthMat[3] * dx + orthMat[4] * dy + orthMat[5] * dz;
    const cz = orthMat[6] * dx + orthMat[7] * dy + orthMat[8] * dz;
    return Math.sqrt(cx * cx + cy * cy + cz * cz);
}

/* ------------------------------------------------------------------
   Lattice reduction, for the minimum-image convention

   THE PROBLEM

   `d = d - round(d)` in fractional coordinates is the minimum-image
   convention every molecular-dynamics code uses, and in a skewed cell it is
   wrong. Independent rounding lands in the box [-1/2, 1/2)^3, and while that
   box holds exactly one representative of every lattice coset, it is not the
   set of SHORTEST representatives. In a monoclinic cell at beta = 120 the box
   corner sits far outside the Wigner-Seitz cell, and a pair of atoms whose
   true separation is 1.8 A through a cell face can be reported at 3.4 A.

   In a Patterson search that is not a cosmetic error. The same routine decides
   the clash penalty and the distance-window restraint, so a missed short
   contact is a physically impossible structure that the swarm is never told
   about, competing on correlation alone against the right answer.

   THE FIX, AND WHY IT IS FREE

   The box does contain the ball of radius w_min/2, where w_min is the smallest
   perpendicular width of the cell. Any separation shorter than that has its
   shortest representative inside the ball, hence inside the box, and since the
   box holds one representative per coset, independent rounding must find it.
   So naive rounding is EXACT below w_min/2 and only unreliable above it.

   w_min is tiny for a skewed basis and large for a compact one, and every
   lattice has a compact basis describing it. Reducing the basis therefore
   raises the radius over which rounding is exact - typically well past any
   contact distance - at no per-distance cost at all. The alternative, testing
   all 27 neighbouring translations inside an O(N^2) loop, costs 27x in the
   hottest loop the kernel has.

   The change of basis is an integer matrix of determinant +/-1, so its inverse
   is integer too and the coordinate transform is exact.
   ------------------------------------------------------------------ */

/** 3x3 integer determinant. */
function _sharkoDet3(m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/** Inverse of an integer 3x3 with |det| = 1, exactly, as integers. */
function _sharkoIntInverse3(m) {
    const det = _sharkoDet3(m);
    if (Math.abs(det) !== 1) return null;
    const adj = [
        (m[4] * m[8] - m[5] * m[7]), -(m[1] * m[8] - m[2] * m[7]),  (m[1] * m[5] - m[2] * m[4]),
       -(m[3] * m[8] - m[5] * m[6]),  (m[0] * m[8] - m[2] * m[6]), -(m[0] * m[5] - m[2] * m[3]),
        (m[3] * m[7] - m[4] * m[6]), -(m[0] * m[7] - m[1] * m[6]),  (m[0] * m[4] - m[1] * m[3])
    ];
    return adj.map(v => v / det);
}

/**
 * A reduced basis for the same lattice, with the change of basis and the
 * radius below which independent rounding is provably the nearest image.
 *
 * The reduction is iterated pairwise Gauss reduction - subtract from each
 * vector the nearest integer multiple of each other vector, sort by length,
 * repeat - followed by a small search over the face diagonals of the two
 * shortest vectors, which is what pairwise steps alone can miss. This is not
 * certified Minkowski reduction and does not need to be: `safeRadius` is
 * measured from the basis that actually came out, so the guarantee holds
 * whatever the reduction achieved.
 *
 * @returns {{ orth, xform, safeRadius, widths, changed, M }}
 *   orth       fractional(reduced) -> Cartesian, row-major 3x3
 *   xform      fractional(original) -> fractional(reduced), integer, row-major
 *   safeRadius half the smallest perpendicular width of the reduced cell, in A
 *   changed    false when the original basis was already reduced
 */
function sharkoReducedCell(cell) {
    const orth = sharkoOrthMatrix(cell);
    // Rows of A are the lattice vectors: A = transpose(orth), because orth maps
    // a fractional COLUMN to Cartesian.
    let v = [
        [orth[0], orth[3], orth[6]],
        [orth[1], orth[4], orth[7]],
        [orth[2], orth[5], orth[8]]
    ];
    // M expresses the current basis in the original one; it starts as identity
    // and only ever gains integer multiples of its own rows, so det stays +/-1.
    let M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const nrm = a => dot(a, a);
    const axpy = (i, j, mu) => {                 // row_i -= mu * row_j
        for (let k = 0; k < 3; k++) { v[i][k] -= mu * v[j][k]; M[i][k] -= mu * M[j][k]; }
    };
    const sortByLength = () => {
        const order = [0, 1, 2].sort((a, b) => nrm(v[a]) - nrm(v[b]));
        v = order.map(i => v[i]); M = order.map(i => M[i]);
    };

    for (let pass = 0; pass < 64; pass++) {
        let moved = false;
        sortByLength();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                if (i === j) continue;
                const nj = nrm(v[j]);
                if (!(nj > 1e-12)) continue;
                const mu = Math.round(dot(v[i], v[j]) / nj);
                if (mu !== 0) { axpy(i, j, mu); moved = true; }
            }
        }
        // Face diagonals: b3 +/- b1 +/- b2 can be shorter than b3 even when b3
        // is already size-reduced against b1 and b2 separately.
        sortByLength();
        let best = null, bestN = nrm(v[2]);
        for (let c1 = -1; c1 <= 1; c1++) {
            for (let c2 = -1; c2 <= 1; c2++) {
                if (!c1 && !c2) continue;
                const t = [0, 1, 2].map(k => v[2][k] + c1 * v[0][k] + c2 * v[1][k]);
                const n = nrm(t);
                if (n < bestN - 1e-9) { bestN = n; best = [c1, c2]; }
            }
        }
        if (best) {
            for (let k = 0; k < 3; k++) {
                v[2][k] += best[0] * v[0][k] + best[1] * v[1][k];
                M[2][k] += best[0] * M[0][k] + best[1] * M[1][k];
            }
            moved = true;
        }
        if (!moved) break;
    }

    const Mflat = [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2]];
    const changed = Mflat.some((x, i) => x !== [1, 0, 0, 0, 1, 0, 0, 0, 1][i]);

    // orth' = orth . transpose(M), since A' = M.A and orth = transpose(A).
    const orthR = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            let sum = 0;
            for (let k = 0; k < 3; k++) sum += orth[r * 3 + k] * Mflat[c * 3 + k];
            orthR[r * 3 + c] = sum;
        }
    }

    // f' = transpose(inverse(M)) . f for a fractional COLUMN vector.
    const Minv = _sharkoIntInverse3(Mflat);
    if (!Minv) {
        // Cannot happen for a determinant-preserving reduction, but a silent
        // wrong answer here would be untraceable, so fall back to no change.
        return { orth: Float64Array.from(orth), xform: new Float64Array([1,0,0,0,1,0,0,0,1]),
                 safeRadius: _sharkoSafeRadius(orth), widths: _sharkoWidths(orth),
                 changed: false, M: [1,0,0,0,1,0,0,0,1], degenerate: true };
    }
    const xform = new Float64Array([Minv[0], Minv[3], Minv[6],
                                    Minv[1], Minv[4], Minv[7],
                                    Minv[2], Minv[5], Minv[8]]);

    const widths = _sharkoWidths(orthR);
    return { orth: orthR, xform, widths, safeRadius: Math.min(...widths) / 2,
             changed, M: Mflat };
}

/** Perpendicular widths of a cell given its fractional->Cartesian matrix. */
function _sharkoWidths(o) {
    const a = [o[0], o[3], o[6]], b = [o[1], o[4], o[7]], c = [o[2], o[5], o[8]];
    const cross = (p, q) => [p[1]*q[2] - p[2]*q[1], p[2]*q[0] - p[0]*q[2], p[0]*q[1] - p[1]*q[0]];
    const len = p => Math.hypot(p[0], p[1], p[2]);
    const bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const V = Math.abs(a[0]*bc[0] + a[1]*bc[1] + a[2]*bc[2]);
    return [V / len(bc), V / len(ca), V / len(ab)];
}

function _sharkoSafeRadius(o) { return Math.min(..._sharkoWidths(o)) / 2; }

/**
 * Cartesian nearest-image distance, correct for any cell.
 *
 * Takes a fractional difference in the ORIGINAL basis and a reduction from
 * sharkoReducedCell(). Exact for separations below `red.safeRadius`; above it
 * the answer is an upper bound, which is all any caller needs since every
 * threshold in this application sits far below the radius.
 */
function sharkoMinImageDistance(dx, dy, dz, red) {
    const X = red.xform;
    let u = X[0]*dx + X[1]*dy + X[2]*dz;
    let v = X[3]*dx + X[4]*dy + X[5]*dz;
    let w = X[6]*dx + X[7]*dy + X[8]*dz;
    u -= Math.round(u); v -= Math.round(v); w -= Math.round(w);
    const o = red.orth;
    const cx = o[0]*u + o[1]*v + o[2]*w;
    const cy = o[3]*u + o[4]*v + o[5]*w;
    const cz = o[6]*u + o[7]*v + o[8]*w;
    return Math.sqrt(cx*cx + cy*cy + cz*cz);
}

/**
 * Reciprocal-space matrix B (row-major 3x3) with s_cartesian = B . (h,k,l),
 * i.e. B = transpose(inverse(M)) for the fractional->Cartesian matrix M.
 * |B.h| is 1/d(hkl), so this is what sets the resolution limit of the data
 * and therefore the width of every peak in the Patterson map.
 */
function sharkoReciprocalMatrix(cell) {
    const M = sharkoOrthMatrix(cell);
    const det = M[0]*(M[4]*M[8] - M[5]*M[7])
              - M[1]*(M[3]*M[8] - M[5]*M[6])
              + M[2]*(M[3]*M[7] - M[4]*M[6]);
    if (!isFinite(det) || Math.abs(det) < 1e-12) return null;
    const id = 1 / det;
    // inverse(M), row-major
    const inv = [
        (M[4]*M[8] - M[5]*M[7]) * id, (M[2]*M[7] - M[1]*M[8]) * id, (M[1]*M[5] - M[2]*M[4]) * id,
        (M[5]*M[6] - M[3]*M[8]) * id, (M[0]*M[8] - M[2]*M[6]) * id, (M[2]*M[3] - M[0]*M[5]) * id,
        (M[3]*M[7] - M[4]*M[6]) * id, (M[1]*M[6] - M[0]*M[7]) * id, (M[0]*M[4] - M[1]*M[3]) * id
    ];
    // transpose
    return new Float64Array([inv[0], inv[3], inv[6],
                             inv[1], inv[4], inv[7],
                             inv[2], inv[5], inv[8]]);
}

/** 1/d(hkl) in A^-1, given the B matrix from sharkoReciprocalMatrix(). */
function sharkoDStar(h, k, l, B) {
    const x = B[0]*h + B[1]*k + B[2]*l;
    const y = B[3]*h + B[4]*k + B[5]*l;
    const z = B[6]*h + B[7]*k + B[8]*l;
    return Math.sqrt(x*x + y*y + z*z);
}

/**
 * Reciprocal-axis lengths |a*|, |b*|, |c*| in A^-1. A step of one unit in the
 * fractional coordinate u moves a point by 1/|a*| Angstroms along the normal
 * to (100), which is what converts a Cartesian peak width into a per-axis
 * width on the fractional grid.
 */
function sharkoReciprocalAxisLengths(cell) {
    const B = sharkoReciprocalMatrix(cell);
    if (!B) return null;
    return [sharkoDStar(1, 0, 0, B), sharkoDStar(0, 1, 0, B), sharkoDStar(0, 0, 1, B)];
}

/* ------------------------------------------------------------------ */
/*  FFT                                                                */
/* ------------------------------------------------------------------ */

/**
 * Smallest power of two >= n, capped at SHARKO_MAX_FFT_SIZE.
 *
 * Radix-2 keeps the transform short and dependable; the cost of rounding a
 * requested 50 up to 64 is nothing next to what the transform saves. The cap
 * exists because the two Float64Array working buffers are 16 bytes per voxel,
 * so 256^3 already costs ~270 MB and anything larger is not worth attempting
 * in a browser worker.
 */
const SHARKO_MAX_FFT_SIZE = 256;
function sharkoNextFFTSize(n) {
    let s = 8;
    while (s < n && s < SHARKO_MAX_FFT_SIZE) s <<= 1;
    return s;
}

/** In-place radix-2 Cooley-Tukey on one line. Twiddles come from a table so
 *  the error does not accumulate the way recurrent rotation does. */
function _sharkoFFT1D(re, im, n, sign, cosT, sinT) {
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const half = len >> 1, step = n / len;
        for (let i = 0; i < n; i += len) {
            for (let k = 0; k < half; k++) {
                const tw = k * step;
                const wr = cosT[tw], wi = sign * sinT[tw];
                const a = i + k, b = a + half;
                const vr = re[b] * wr - im[b] * wi;
                const vi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - vr; im[b] = im[a] - vi;
                re[a] += vr;        im[a] += vi;
            }
        }
    }
}

/**
 * In-place unnormalised 3D FFT of an N^3 complex grid indexed
 * idx = iw*N*N + iv*N + iu. sign = +1 gives the inverse (synthesis)
 * convention f(x) = sum_k F(k) exp(+2*pi*i*k.x/N).
 */
function sharkoFFT3D(re, im, N, sign) {
    const cosT = new Float64Array(N), sinT = new Float64Array(N);
    for (let i = 0; i < N; i++) {
        cosT[i] = Math.cos(2 * Math.PI * i / N);
        sinT[i] = Math.sin(2 * Math.PI * i / N);
    }
    const lr = new Float64Array(N), li = new Float64Array(N);
    const N2 = N * N;

    const runAxis = (stride, outerA, outerB) => {
        for (let a = 0; a < N; a++) {
            for (let b = 0; b < N; b++) {
                const base = a * outerA + b * outerB;
                for (let i = 0; i < N; i++) { const p = base + i * stride; lr[i] = re[p]; li[i] = im[p]; }
                _sharkoFFT1D(lr, li, N, sign, cosT, sinT);
                for (let i = 0; i < N; i++) { const p = base + i * stride; re[p] = lr[i]; im[p] = li[i]; }
            }
        }
    };

    runAxis(1,  N2, N);   // along u
    runAxis(N,  N2, 1);   // along v
    runAxis(N2, N,  1);   // along w
}

/* ------------------------------------------------------------------ */
/*  Patterson synthesis                                                */
/* ------------------------------------------------------------------ */


/**
 * Patterson map by FFT.
 *
 *   P(u,v,w) = (1/V) * sum_hkl I(hkl) * exp(2*pi*i*(hu + kv + lw))
 *
 * With u = iu/N this is exactly an unnormalised inverse DFT of the grid that
 * holds I(hkl) at (h mod N, k mod N, l mod N), so the whole map costs
 * O(N^3 log N) instead of the O(N^3 * numReflections) the direct summation
 * cost. At N=64 with 32000 expanded reflections that is roughly four million
 * operations in place of eight billion.
 *
 * The expanded list contains every Friedel mate, so I(h) = I(-h) and the
 * transform is Hermitian: the imaginary part cancels and the map is real.
 * A residual imaginary component is reported as a warning because it can only
 * come from an expansion that was not centrosymmetric.
 *
 * ALIASING: a reflection at index h and one at h+N land on the same grid
 * point, so the grid must satisfy N >= 2*hmax+1 for the synthesis to be exact.
 * The requested resolution is raised to meet that; if even the FFT size cap
 * cannot, the offending high-order reflections are dropped and a warning is
 * raised rather than being silently folded back on top of low-order data.
 *
 * Returns { map, res, dMin, sigma, hmax, warnings }.
 */
function sharkoPattersonFFT(fullReflections, cell, requestedRes, lorchStrength = 0) {
    const V = sharkoCellVolume(cell);
    if (!V || !isFinite(V) || V <= 0) throw new Error(`Invalid cell volume: ${V}`);

    const warnings = [];
    const B = sharkoReciprocalMatrix(cell);
    if (!B) throw new Error('Cell is degenerate: the orthogonalisation matrix is not invertible.');

    // Resolution limit of the data, which sets the peak width.
    let dStarMax = 0, hmax = 0;
    for (const r of fullReflections) {
        if (!Number.isFinite(r.h) || !Number.isFinite(r.k) || !Number.isFinite(r.l)) continue;
        if (!Number.isFinite(r.intensity)) continue;
        const ds = sharkoDStar(r.h, r.k, r.l, B);
        if (ds > dStarMax) dStarMax = ds;
        hmax = Math.max(hmax, Math.abs(r.h), Math.abs(r.k), Math.abs(r.l));
    }
    if (hmax === 0) throw new Error('No usable reflections after expansion.');

    const needed = 2 * hmax + 1;
    const N = sharkoNextFFTSize(Math.max(requestedRes || 0, needed));
    if (N < needed) {
        warnings.push(
            `The data reach index ${hmax}, which needs a ${needed}^3 grid to transform without ` +
            `aliasing, but the FFT size is capped at ${SHARKO_MAX_FFT_SIZE}. Reflections with any ` +
            `index above ${Math.floor((N - 1) / 2)} have been dropped instead of being folded back ` +
            `on top of low-order data.`);
    } else if (N > (requestedRes || 0)) {
        warnings.push(
            `Map grid raised from the requested ${requestedRes} to ${N} - the FFT needs a ` +
            `power-of-two grid of at least ${needed} to represent index ${hmax} without aliasing.`);
    }

    const limit = Math.floor((N - 1) / 2);
    const N3 = N * N * N;
    const re = new Float64Array(N3);
    const im = new Float64Array(N3);

    // Derived BEFORE accumulation so we can apply the Lorch parameter immediately.
    const dMin = dStarMax > 0 ? 1 / dStarMax : NaN;

    let placed = 0, dropped = 0;
    for (const r of fullReflections) {
        const h = r.h, k = r.k, l = r.l;
        let I = r.intensity;
        if (!Number.isFinite(h) || !Number.isFinite(k) || !Number.isFinite(l) || !Number.isFinite(I)) continue;
        if (Math.abs(h) > limit || Math.abs(k) > limit || Math.abs(l) > limit) { dropped++; continue; }

        if (lorchStrength > 0 && Number.isFinite(dMin)) {
            const ds = sharkoDStar(h, k, l, B);
            const arg = Math.PI * ds * dMin;
            const sinc = arg === 0 ? 1 : Math.sin(arg) / arg;
            I *= (1 - lorchStrength) + (lorchStrength * sinc);
        }

        const ih = ((h % N) + N) % N;
        const ik = ((k % N) + N) % N;
        const il = ((l % N) + N) % N;
        re[il * N * N + ik * N + ih] += I;   // accumulate: never overwrite
        placed++;
    }
    if (!placed) throw new Error('Every reflection fell outside the transform grid.');
    if (dropped) warnings.push(`${dropped} reflection(s) beyond index ${limit} were dropped (see above).`);

    sharkoFFT3D(re, im, N, +1);

    // Hermitian input means a real result; anything else signals a
    // non-centrosymmetric expansion and would make the map meaningless.
    let maxRe = 0, maxIm = 0;
    for (let i = 0; i < N3; i++) {
        const ar = Math.abs(re[i]); if (ar > maxRe) maxRe = ar;
        const ai = Math.abs(im[i]); if (ai > maxIm) maxIm = ai;
    }
    if (maxRe > 0 && maxIm / maxRe > 1e-4) {
        warnings.push(
            `The transform left an imaginary component of ${(100 * maxIm / maxRe).toFixed(2)}% of the ` +
            `map maximum. A Patterson synthesis over a Friedel-complete list should be exactly real, ` +
            `so the reflection expansion is probably not centrosymmetric.`);
    }

    const map = new Float32Array(N3);
    for (let i = 0; i < N3; i++) map[i] = re[i] / V;

    // Peak width. A Fourier series truncated at a sphere of radius 1/dMin has
    // a point-spread function whose FWHM is about 0.61*dMin; matching a
    // Gaussian to that FWHM gives sigma = 0.61*dMin/2.3548 ~ 0.26*dMin. The
    // floor keeps the peak at least comparable to the grid spacing, since a
    // Gaussian narrower than the sampling cannot be represented at all.
    const axes = sharkoReciprocalAxisLengths(cell) || [1, 1, 1];
    const coarsestSpacing = Math.max(1 / (axes[0] * N), 1 / (axes[1] * N), 1 / (axes[2] * N));
    const sigma = Math.max(isFinite(dMin) ? 0.26 * dMin : 0, 0.7 * coarsestSpacing);

    return { map, res: N, dMin, sigma, hmax, warnings };
}


/* ------------------------------------------------------------------ */
/*  Lorentz-polarisation                                               */
/* ------------------------------------------------------------------ */

/**
 * Lorentz-polarisation factor, written in exactly the form Powder 5 states in
 * the "--- Lorentz-Polarisation ---" block of its report:
 *
 *   Lp(2th) = L(2th) * P(2th)
 *     L(2th) = 1 / ( sin^2(th) * cos(th) )
 *     P(2th) = ( 1 + K * cos^2(2th) ) / ( 1 + K )
 *
 * K is the polarisation constant: 1 for an unpolarised beam (lab tube, no
 * monochromator), cos^2(2theta_m) for a monochromated beam. The report writes
 * it out as "Polarisation K", so it is read from the file rather than assumed.
 *
 * This is only a fallback. When the file supplies its own Lp column that value
 * is used instead, since it was computed at full precision from the refined
 * 2theta rather than from the value rounded for printing.
 */
function sharkoLorentzPolarization(tthDeg, K) {
    if (!Number.isFinite(tthDeg) || tthDeg <= 0 || tthDeg >= 180) return 1;
    const d2r = Math.PI / 180;
    const th = 0.5 * tthDeg * d2r;
    const s = Math.sin(th), c = Math.cos(th);
    if (s < 1e-6 || c < 1e-6) return 1;
    const k = Number.isFinite(K) && K >= 0 ? K : 1;
    const P = (1 + k * Math.cos(tthDeg * d2r) ** 2) / (1 + k);
    return P / (s * s * c);
}

/* ------------------------------------------------------------------ */
/*  Space-group setting resolution                                     */
/* ------------------------------------------------------------------ */

/**
 * The JSON keys settings by symbol (e.g. "P2122"), not by a flat "rotations"
 * field. This normalises a symbol string for matching (strips whitespace,
 * uppercases) so a parsed name like "P 21 2 2" matches a JSON symbol like
 * "P2122".
 */
function normalizeSGSymbol(s) {
    return (s || '').replace(/\s+/g, '').toUpperCase();
}

/**
 * Picks the entry from spaceGroups[sgNumber].settings[] matching the crystal's
 * actual setting, and reports how confident that match is.
 *
 * Returns { setting, matched, warnings[] }.
 *   matched: 'symbol'  - the setting symbol matched the name in the file
 *            'only'    - the space group has exactly one setting, no ambiguity
 *            'default' - no match; the first listed setting was used (WARNING)
 * Returns null if the space group number itself is not in the JSON.
 */
function resolveSpaceGroupSetting(spaceGroups, sgNumber, sgNameRaw) {
    if (!spaceGroups) return null;
    const entry = spaceGroups[sgNumber] || spaceGroups[String(sgNumber)];
    if (!entry) return null;

    const settings = entry.settings || [entry]; // tolerate a flat legacy shape
    if (!settings.length) return null;

    const warnings = [];
    const target = normalizeSGSymbol(sgNameRaw);

    if (target) {
        const match = settings.find(s => normalizeSGSymbol(s.symbol) === target);
        if (match) return { setting: match, matched: 'symbol', warnings };
    }

    if (settings.length === 1) {
        if (target && normalizeSGSymbol(settings[0].symbol) !== target) {
            warnings.push(`The file names space group ${sgNumber} as "${sgNameRaw}", but the ` +
                          `database lists it only as "${settings[0].symbol}". Using that setting.`);
        }
        return { setting: settings[0], matched: 'only', warnings };
    }

    warnings.push(`Space group ${sgNumber} has ${settings.length} settings in the database ` +
                  `(${settings.map(s => s.symbol).join(', ')}) and none matches the name ` +
                  `"${sgNameRaw || '(none)'}" from the file. Falling back to "${settings[0].symbol}". ` +
                  `If that is the wrong setting every orbit, Harker section and site will be wrong.`);
    return { setting: settings[0], matched: 'default', warnings };
}

/** Backwards-compatible wrapper: returns just the setting object, or null. */
function findSpaceGroupSetting(spaceGroups, sgNumber, sgNameRaw) {
    const r = resolveSpaceGroupSetting(spaceGroups, sgNumber, sgNameRaw);
    return r ? r.setting : null;
}

/**
 * Extracts the rotation matrices to expand reflections with, and says where
 * they came from. Reflection indices transform by the rotation part only, so
 * translations are irrelevant here - but their PRESENCE is not, because a
 * setting carrying full sym_ops is the one that also lets the swarm build
 * structures and lets systematic absences be reasoned about.
 *
 * Returns { rotations, source, opCount, warnings[] } or null if the setting
 * carries no usable operators at all.
 *   source: 'sym_ops'   - full operator list (rotations + translations)
 *           'rotations' - legacy rotation-only list, no translations
 */
function getExpansionOperators(setting) {
    if (!setting) return null;
    const warnings = [];

    const symOps = Array.isArray(setting.sym_ops) ? setting.sym_ops : null;
    if (symOps && symOps.length) {
        const rotations = symOps.map(op => op && op.r).filter(r => Array.isArray(r) && r.length === 9);
        if (rotations.length !== symOps.length) {
            warnings.push(`${symOps.length - rotations.length} of ${symOps.length} operators in ` +
                          `sym_ops for "${setting.symbol || 'this setting'}" have a malformed ` +
                          `rotation matrix and were skipped.`);
        }
        if (rotations.length) {
            const orderZ = Number(setting.order_z);
            if (Number.isFinite(orderZ) && orderZ > 0 && symOps.length !== orderZ) {
                warnings.push(`sym_ops for "${setting.symbol || 'this setting'}" holds ` +
                              `${symOps.length} operator(s) but order_z is ${orderZ}. The operator ` +
                              `list in the JSON is incomplete; orbit sizes will be wrong.`);
            }
            return { rotations, source: 'sym_ops', opCount: symOps.length, warnings };
        }
    }

    // Fall back to the rotation-only list the generator still writes.
    const rots = Array.isArray(setting.rotations) ? setting.rotations : null;
    if (rots && rots.length) {
        const rotations = rots.map(r => Array.isArray(r) ? r : Array.from(r || []))
                              .filter(r => r.length === 9);
        if (rotations.length) {
            warnings.push(`Space-group setting "${setting.symbol || '?'}" has no sym_ops in the ` +
                          `database, only the legacy rotation-only list. Orbit sizes are still ` +
                          `correct, but there are no translations: screw/glide absences cannot be ` +
                          `checked and the swarm cannot build symmetry-equivalent atoms. ` +
                          `Re-run cctbx_generate_sg_harker_v6.py with a cctbx build that exposes ` +
                          `the operator accessor.`);
            return { rotations, source: 'rotations', opCount: rotations.length, warnings };
        }
    }

    return null;
}

/**
 * Expands a unique reflection list to the full sphere using the matched
 * setting's operators (rotation parts only - reflection indices don't
 * transform by translation) plus the Friedel pair (-h,-k,-l).
 *
 * The Pawley/Le Bail intensity of a powder reflection is the sum over its
 * whole orbit, so it is divided by the orbit size and spread over the orbit;
 * the total is conserved exactly.
 *
 * Returns { reflections, symmetry } where symmetry is
 *   { ok, source, opCount, settingSymbol, matched, description, warnings[] }
 * and throws if no usable symmetry information exists at all - a silent
 * identity-only fallback would produce an orbit of 2 for every reflection and
 * a map that looks plausible while being wrong.
 */
function expandReflections(uniqueReflections, sgNumber, sgNameRaw, spaceGroups, options) {
    const opts = options || {};
    // Two possible meanings for ref.intensity, and they scale differently:
    //
    //  perReflection === false (default): the value is the whole orbit's
    //      intensity, m.Lp.|F|^2 as a powder measures it. The multiplicity is
    //      absorbed by dividing by the orbit size and spreading it over the
    //      orbit, so the total is conserved.
    //
    //  perReflection === true: the value is already |F|^2 for a SINGLE
    //      reflection, because the file supplied |Fo| (Powder 5 having divided
    //      out m and Lp itself). Every member of the orbit then carries that
    //      same value and nothing is divided - dividing again would suppress
    //      high-multiplicity reflections by up to a factor of 8.
    const perReflection = !!opts.perReflection;

    const resolved = resolveSpaceGroupSetting(spaceGroups, sgNumber, sgNameRaw);
    if (!resolved) {
        throw new Error(
            `Space group ${sgNumber ?? '?'} ("${sgNameRaw || 'unnamed'}") is not in the symmetry ` +
            `database (sg/). Reflections cannot be expanded ` +
            `to the full sphere, so no Patterson map can be computed.`);
    }

    const ops = getExpansionOperators(resolved.setting);
    if (!ops) {
        throw new Error(
            `Space-group setting "${resolved.setting.symbol || sgNameRaw || sgNumber}" carries ` +
            `neither sym_ops nor rotations in the symmetry database. Expanding with the identity ` +
            `alone would give every reflection an orbit of 2 and silently produce a wrong map, ` +
            `so the calculation is stopped instead. Regenerate the JSON with ` +
            `cctbx_generate_sg_harker_v6.py.`);
    }

    const warnings = resolved.warnings.concat(ops.warnings);
    const rotations = ops.rotations;
    const fullSphere = new Map();
    let minOrbit = Infinity, maxOrbit = 0;
    const multMismatches = [];

    uniqueReflections.forEach(ref => {
        const h = ref.h, k = ref.k, l = ref.l;
        const I = ref.intensity;
        if (!Number.isFinite(h) || !Number.isFinite(k) || !Number.isFinite(l) || !Number.isFinite(I)) return;

        const equivalents = new Map();
        rotations.forEach(R => {
            // Miller indices transform as a row vector: h' = h R.
            // R is row-major with x'_i = sum_j R_ij x_j, so h'_j = sum_i h_i R_ij.
            const hp = Math.round(h * R[0] + k * R[3] + l * R[6]);
            const kp = Math.round(h * R[1] + k * R[4] + l * R[7]);
            const lp = Math.round(h * R[2] + k * R[5] + l * R[8]);
            equivalents.set(`${hp},${kp},${lp}`, { h: hp, k: kp, l: lp });
            equivalents.set(`${-hp},${-kp},${-lp}`, { h: -hp, k: -kp, l: -lp });
        });

        // Multiplicity of this reflection's orbit, Friedel pairs included.
        const multiplicity = equivalents.size;
        if (multiplicity < minOrbit) minOrbit = multiplicity;
        if (multiplicity > maxOrbit) maxOrbit = multiplicity;

        // The file now states the powder multiplicity Powder 5 used. If our
        // orbit disagrees with it, the two programs are using different
        // symmetry - almost always a wrong setting - and every intensity is
        // scaled wrongly. This is the sharpest available check on the
        // space-group setting, so it is worth making noisily.
        if (Number.isFinite(ref.multiplicity) && ref.multiplicity > 0 && ref.multiplicity !== multiplicity) {
            multMismatches.push(`(${h},${k},${l}): file m=${ref.multiplicity}, orbit=${multiplicity}`);
        }

        const intensityPerRef = perReflection ? I : I / multiplicity;

        equivalents.forEach((val, key) => {
            const existing = fullSphere.get(key);
            if (existing) {
                // Two entries landing on the same index means the input list
                // held a duplicate hkl. With orbit-summed intensities the
                // contributions add; with per-reflection |F|^2 they are two
                // estimates of one quantity, so they are averaged.
                existing.merged = (existing.merged || 1) + 1;
                if (perReflection) {
                    existing.intensity += (intensityPerRef - existing.intensity) / existing.merged;
                } else {
                    existing.intensity += intensityPerRef;
                }
            } else {
                fullSphere.set(key, { h: val.h, k: val.k, l: val.l, intensity: intensityPerRef });
            }
        });
    });

    const merged = Array.from(fullSphere.values()).filter(r => r.merged).length;
    if (merged) {
        warnings.push(`${merged} expanded reflection(s) received contributions from more than one ` +
                      `entry in the input list (duplicate hkl). They were ` +
                      `${perReflection ? 'averaged' : 'summed'}.`);
    }
    if (multMismatches.length) {
        warnings.push(`The multiplicity in the file disagrees with the orbit computed from ` +
                      `"${resolved.setting.symbol}" for ${multMismatches.length} of ` +
                      `${uniqueReflections.length} reflection(s), e.g. ${multMismatches.slice(0, 3).join('; ')}. ` +
                      `The two programs are not using the same symmetry - check the space-group ` +
                      `setting before trusting the map.`);
    }

    const symmetry = {
        ok: true,
        source: ops.source,
        opCount: ops.opCount,
        settingSymbol: resolved.setting.symbol || String(sgNumber),
        matched: resolved.matched,
        orbitRange: maxOrbit ? [minOrbit, maxOrbit] : null,
        expandedCount: fullSphere.size,
        perReflection,
        multiplicityChecked: uniqueReflections.some(r => Number.isFinite(r.multiplicity)),
        multiplicityMismatches: multMismatches.length,
        warnings,
        description: ops.source === 'sym_ops'
            ? `full operators (${ops.opCount}) of ${resolved.setting.symbol || sgNumber}`
            : `rotations only (${ops.opCount}, no translations) of ${resolved.setting.symbol || sgNumber}`
    };

    return { reflections: Array.from(fullSphere.values()), symmetry };
}

/* ------------------------------------------------------------------ */
/*  Gaussian smearing                                                  */
/* ------------------------------------------------------------------ */

/**
 * Convolves an N^3 map in place with a Gaussian of standard deviation sigma
 * Angstroms, wrapping at the cell edges because the Patterson function is
 * periodic.
 *
 * A model map built by dropping each interatomic vector into the single
 * nearest voxel is a set of delta functions, but the observed map it is being
 * compared against is resolution-limited: every one of its peaks is about
 * 0.6*dMin wide. Without this smearing the calculated map cannot resemble the
 * observed one at any scale factor, and the difference map is just the
 * observed map with a few pinholes punched in it.
 *
 * The kernel is applied as three 1D passes with a per-axis width taken from
 * the reciprocal-axis lengths, so it is separable. That is exact for
 * orthogonal cells; for oblique cells the true Cartesian Gaussian picks up
 * cross terms this ignores, which slightly rounds the ellipsoid. The swarm
 * fitness never uses this approximation - it works from exact Cartesian
 * distances - so the effect is confined to the displayed map.
 */
function sharkoSmearMap(map, N, sigma, cell, maxRadius) {
    const axes = sharkoReciprocalAxisLengths(cell);
    if (!axes || !(sigma > 0)) return map;
    // Half the grid is the most a periodic kernel can ever span; beyond that
    // it wraps onto itself.
    const cap = Number.isFinite(maxRadius) ? maxRadius : Math.floor(N / 2);
    const line = new Float32Array(N);
    const N2 = N * N;

    const pass = (stride, outerA, outerB, axisIdx) => {
        // Fractional sigma along this axis: one unit of the fractional
        // coordinate spans 1/|a*| Angstroms perpendicular to the axis planes.
        const sigFrac = sigma * axes[axisIdx];
        const sigVox = sigFrac * N;
        if (!(sigVox > 0.05)) return;                 // narrower than the grid: nothing to do
        // A Gaussian truncated well inside its own width and then renormalised
        // is a box filter, and a separable box filter makes axis-aligned CUBES.
        // The old default cap of 6 voxels did exactly that: on a 64^3 grid at
        // 1 A resolution sigVox is around 4.3, so the kernel was cut at 6
        // voxels, where the Gaussian has only fallen to 0.38 of its peak. Every
        // calculated peak came out square, and the difference map showed square
        // holes where round ones belonged. Cutting at 3 sigma - where the
        // Gaussian is at 0.011 - costs a wider kernel and gives round peaks.
        const rad = Math.min(cap, Math.max(1, Math.ceil(3 * sigVox)));

        const kern = new Float64Array(2 * rad + 1);
        let ksum = 0;
        for (let d = -rad; d <= rad; d++) {
            const g = Math.exp(-(d * d) / (2 * sigVox * sigVox));
            kern[d + rad] = g; ksum += g;
        }
        for (let i = 0; i < kern.length; i++) kern[i] /= ksum;

        for (let a = 0; a < N; a++) {
            for (let b = 0; b < N; b++) {
                const base = a * outerA + b * outerB;
                for (let i = 0; i < N; i++) line[i] = map[base + i * stride];
                for (let i = 0; i < N; i++) {
                    let acc = 0;
                    for (let d = -rad; d <= rad; d++) acc += kern[d + rad] * line[(i + d + N) % N];
                    map[base + i * stride] = acc;
                }
            }
        }
    };

    pass(1,  N2, N, 0);
    pass(N,  N2, 1, 1);
    pass(N2, N,  1, 2);
    return map;
}

/**
 * Least-squares scale factor k minimising sum (obs - k*calc)^2, restricted to
 * the voxels the mask leaves available.
 *
 * The observed map is in electrons^2 per A^3 after the 1/V of the synthesis;
 * the calculated map is a raw sum of Z_i*Z_j. Subtracting one from the other
 * without putting them on a common scale produced a "difference" map that was
 * simply whichever of the two happened to have larger numbers.
 */
function sharkoMapScaleFactor(obs, calc, mask) {
    let num = 0, den = 0;
    const n = Math.min(obs.length, calc.length);
    for (let i = 0; i < n; i++) {
        if (mask && mask[i]) continue;
        const o = obs[i], c = calc[i];
        if (!isFinite(o) || !isFinite(c)) continue;
        num += o * c; den += c * c;
    }
    return den > 1e-30 ? num / den : 0;
}

/* Node/worker/browser interop -------------------------------------- */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        normalizeSGSymbol, findSpaceGroupSetting, resolveSpaceGroupSetting,
        getExpansionOperators, expandReflections,
        sharkoCellVolume, sharkoOrthMatrix, sharkoFracToCartLength,
        sharkoReducedCell, sharkoMinImageDistance,
        sharkoLorentzPolarization,
        sharkoReciprocalMatrix, sharkoDStar, sharkoReciprocalAxisLengths,
        sharkoNextFFTSize, sharkoFFT3D, sharkoPattersonFFT,
        sharkoSmearMap, sharkoMapScaleFactor,
        SHARKO_MAX_FFT_SIZE
    };
}
