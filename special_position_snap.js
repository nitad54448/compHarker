/* ------------------------------------------------------------------
   Special-position snapping for sHarko.

   Problem: the swarm searches a continuous space, so a site whose true
   position lies on a symmetry element is only ever found *near* it -
   0.74 and 0.76 instead of 0.75. Neither the shader's collapse test
   (SAME_SITE_TOL = 0.05 A) nor the anti-bump rule can pull it the rest
   of the way: the exact special position is a measure-zero target.

   Fix: the operators that map a site to within `tolA` of itself are its
   site-symmetry group (its stabiliser). Averaging the site over that
   group projects it onto the group's invariant subspace, which IS the
   special position. For a mirror at x = 3/4 (x -> 3/2 - x), 0.74 and its
   image 0.76 average to 0.75 exactly.

   This is a post-processing / polish step. It does not change the search
   method - it only cleans up the answer the search already gave.
   ------------------------------------------------------------------ */

/**
 * @param {{x:number,y:number,z:number}} p  fractional site
 * @param {Array} symOps                    [{r:[9], t:[3]}, ...]
 * @param {Float64Array|Array} orth         3x3 fractional -> Cartesian
 * @param {number} tolA                     coincidence tolerance, Angstrom
 * @returns {{x,y,z,order,shift}} snapped position, stabiliser order,
 *                                and how far it moved (A)
 */
function snapToSpecialPosition(p, symOps, orth, tolA = 0.7) {
    let cur = { x: p.x, y: p.y, z: p.z };
    let order = 1;

    // Two or three passes: one average is exact for an order-2 stabiliser
    // and converges quickly for higher ones.
    for (let pass = 0; pass < 3; pass++) {
        let sx = 0, sy = 0, sz = 0, n = 0;
        for (const op of symOps) {
            let nx = cur.x*op.r[0] + cur.y*op.r[1] + cur.z*op.r[2] + op.t[0];
            let ny = cur.x*op.r[3] + cur.y*op.r[4] + cur.z*op.r[5] + op.t[1];
            let nz = cur.x*op.r[6] + cur.y*op.r[7] + cur.z*op.r[8] + op.t[2];

            // Take the image's NEAREST lattice copy to `cur`, not its copy in
            // [0,1). Averaging 0.02 with 0.98 across the cell edge would give
            // 0.50, which is nonsense; shifting to -0.02 first gives 0.00.
            nx -= Math.round(nx - cur.x);
            ny -= Math.round(ny - cur.y);
            nz -= Math.round(nz - cur.z);

            if (sharkoFracToCartLength(nx - cur.x, ny - cur.y, nz - cur.z, orth) > tolA) continue;
            sx += nx; sy += ny; sz += nz; n++;
        }
        if (n <= 1) break;              // general position, nothing to snap to
        order = n;
        cur = { x: sx/n, y: sy/n, z: sz/n };
    }

    const shift = sharkoFracToCartLength(cur.x - p.x, cur.y - p.y, cur.z - p.z, orth);

    // Rational coordinates land on 1/4, 1/3, 1/2... but float averaging leaves
    // 0.7499999. Round anything within 1e-4 of a small rational.
    const tidy = v => {
        v -= Math.floor(v);
        for (const d of [2, 3, 4, 6, 8, 12]) {
            const r = Math.round(v * d) / d;
            if (Math.abs(v - r) < 1e-4) return (r + 1) % 1;
        }
        return v;
    };
    return { x: tidy(cur.x), y: tidy(cur.y), z: tidy(cur.z), order, shift };
}

/**
 * Is site j just site i seen through a symmetry operator?
 *
 * The swarm's fitness is invariant under relabelling, so two "independent"
 * atoms can settle on positions that are symmetry-equivalent to each other.
 * They are then far apart in the cell, so the anti-bump penalty never fires,
 * yet the model has found one site twice and missed another.
 */
function isSymmetryDuplicate(pi, pj, symOps, orth, tolA = 0.7) {
    for (const op of symOps) {
        const nx = pi.x*op.r[0] + pi.y*op.r[1] + pi.z*op.r[2] + op.t[0];
        const ny = pi.x*op.r[3] + pi.y*op.r[4] + pi.z*op.r[5] + op.t[1];
        const nz = pi.x*op.r[6] + pi.y*op.r[7] + pi.z*op.r[8] + op.t[2];
        let dx = nx - pj.x, dy = ny - pj.y, dz = nz - pj.z;
        dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
        if (sharkoFracToCartLength(dx, dy, dz, orth) < tolA) return true;
    }
    return false;
}

/* ------------------------------------------------------------------
   HOOK
   In renderFinalResults(), replace the raw coords with snapped ones
   BEFORE calculateMultiplicity() is called, so the multiplicity and the
   cell mass are computed from the corrected site:

        let px = coords[cIdx*3], py = coords[cIdx*3+1], pz = coords[cIdx*3+2];
        if (symOps && orthMat) {
            const s = snapToSpecialPosition({x:px,y:py,z:pz}, symOps, orthMat);
            if (s.order > 1) {
                console.log(`${a.element}: snapped ${s.shift.toFixed(2)} A onto a `
                          + `site of order ${s.order}`);
                px = s.x; py = s.y; pz = s.z;
            }
        }
        const mult = ... calculateMultiplicity(px, py, pz, symOps, orthMat) ...

   Two things follow from a successful snap:

   1. The site's occupancy is mult / symOps.length, not 1. Write that to
      the CIF _atom_site_occupancy, or the formula sum comes out too high.

   2. calculateMultiplicity()'s tolerance (0.2 A) and the shader's
      SAME_SITE_TOL (0.05 A) both now see an exact coincidence, so the
      collapse works as intended and the images stop being double-counted.

   Optionally, run the same snap on the running best every N generations
   and keep it only if the fitness does not get worse. That turns the
   special position from an unreachable point into an attractor without
   touching the search itself.
   ------------------------------------------------------------------ */

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { snapToSpecialPosition, isSymmetryDuplicate };
}
