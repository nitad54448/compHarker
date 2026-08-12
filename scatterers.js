// Bumped whenever this file changes. Harko.html compares it against what it
// expects and says so if a browser cache is serving something older - a stale
// module reports errors at line numbers that no longer exist, which sends you
// looking for a bug that was already fixed.
const SHARKO_SCATTERERS_VERSION = '2026-08-09j';

/* ------------------------------------------------------------------
   Scattering factors for sHarko.

   Loads the tables written by cctbx_scatterers_v1.py and turns them into the
   formFactor(element, s) callback buildScatteringTable() expects.

   THE ONE THING TO GET RIGHT

   The Gaussian coefficients are defined against stol = sin(theta)/lambda,
   which is HALF of s = 1/d. Passing s where stol belongs leaves f(0)
   untouched - so the mistake survives a casual check - while making every
   atom fall off at four times the correct rate in the exponent. The model
   then looks systematically too diffuse, the correlation drops for reasons
   that have nothing to do with the structure, and the search is quietly
   pulled toward whatever compensates.

   makeFormFactor() therefore takes s and halves it internally, and
   verifyTable() asserts f(0) == Z before anything uses the table.
   ------------------------------------------------------------------ */

const SHARKO_SCATTER_DIR = 'scatters';

/**
 * Fetches the index and the preferred X-ray table.
 *
 * Neutron lengths are loaded too when present: powder work is as often neutron
 * as X-ray, and the two need entirely different scattering. A neutron length
 * has no angular fall-off at all and light atoms are not swamped by heavy
 * ones, so guessing wrong is not a small error.
 */
async function loadScatteringTables(baseDir = SHARKO_SCATTER_DIR) {
    const index = await fetch(`${baseDir}/index.json`).then(r => {
        if (!r.ok) throw new Error(`${baseDir}/index.json: ${r.status}. ` +
            `Run cctbx_scatterers_v1.py to create it.`);
        return r.json();
    });

    const want = index.preferred || 'xray_wk1995.json';
    const xray = await fetch(`${baseDir}/${want}`).then(r => r.ok ? r.json() : null);
    if (!xray) throw new Error(`${baseDir}/${want} is missing.`);

    let neutron = null;
    if ((index.files || []).some(f => f.name === 'neutron.json')) {
        neutron = await fetch(`${baseDir}/neutron.json`).then(r => r.ok ? r.json() : null);
    }
    return { index, xray, neutron };
}

/** f(stol) = c + sum a_i exp(-b_i stol^2). */
function evaluateGaussian(entry, stol) {
    let f = entry.c || 0;
    const a = entry.a, b = entry.b;
    const q = stol * stol;
    for (let i = 0; i < a.length; i++) f += a[i] * Math.exp(-b[i] * q);
    return f;
}

/**
 * Checks f(0) against Z (minus the charge for an ion).
 *
 * Cheap, and it catches the two failures that matter: coefficients pulled from
 * the wrong accessor in the generator, and a table that has been silently
 * truncated. Called once at load; a failure here should stop the run rather
 * than degrade it, because a wrong f is indistinguishable from a wrong
 * structure once it reaches the correlation.
 */
function verifyScatteringTable(xray, tolerance = 0.25) {
    const bad = [];
    for (const [label, e] of Object.entries(xray.table || {})) {
        if (e.Z === undefined) continue;
        const m = /^([A-Za-z]+)(\d+)?([+-])?$/.exec(label);
        let charge = 0;
        if (m && m[2] && m[3]) charge = parseInt(m[2], 10) * (m[3] === '+' ? 1 : -1);
        const got = evaluateGaussian(e, 0);
        if (Math.abs(got - (e.Z - charge)) > tolerance) {
            bad.push(`${label}: expected ${e.Z - charge}, got ${got.toFixed(2)}`);
        }
    }
    return bad;
}

/**
 * The callback buildScatteringTable() wants: (element, s) -> f.
 *
 * @param tables   from loadScatteringTables()
 * @param options  radiation: 'xray' | 'neutron'
 *                 ions: { PB: 'Pb2+', O: 'O2-' } to use ionic factors
 *                 overallB: applied by buildScatteringTable, not here
 *
 * Returns null for an element the table does not hold, which makes
 * buildScatteringTable fall back to the point-atom value and report it, rather
 * than substituting a neighbouring element and saying nothing.
 */
function makeFormFactor(tables, options = {}) {
    const radiation = options.radiation || 'xray';
    const ions = options.ions || {};

    // Table keys are capitalised as in the periodic table ("Pb"), while the
    // app carries uppercase symbols ("PB"). Index both ways once.
    const xrayByKey = new Map();
    for (const [label, e] of Object.entries(tables.xray?.table || {})) {
        xrayByKey.set(label.toUpperCase(), e);
    }
    const neutronByKey = new Map();
    for (const [label, e] of Object.entries(tables.neutron?.table || {})) {
        neutronByKey.set(label.toUpperCase(), e);
    }

    if (radiation === 'neutron') {
        return (element, s) => {
            const e = neutronByKey.get(String(element).toUpperCase());
            // Bound coherent lengths are in femtometres and can be NEGATIVE
            // (H, Ti, Mn). The sign is physical and must be kept: it is what
            // makes contrast variation work, and dropping it would invert
            // those atoms' contribution to every structure factor.
            return e ? e.b_coh : null;
        };
    }

    return (element, s) => {
        const key = String(element).toUpperCase();
        const label = ions[key] ? String(ions[key]).toUpperCase() : key;
        const e = xrayByKey.get(label) || xrayByKey.get(key);
        if (!e) return null;
        return evaluateGaussian(e, s / 2);      // s -> stol
    };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SHARKO_SCATTERERS_VERSION, loadScatteringTables, evaluateGaussian,
                       verifyScatteringTable, makeFormFactor, SHARKO_SCATTER_DIR };
}
