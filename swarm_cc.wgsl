// Map-correlation fitness for sHarko, Wyckoff-constrained, multi-assignment.
//
// Replaces the Patterson vector sum of the retired swarm_multi.wgsl. The fitness is the
// Pearson correlation between the observed and calculated intensities, which
// by Parseval IS the correlation of the two Patterson maps - and is strictly
// better than computing it in real space, because subtracting the mean
// intensity removes the origin peak EXACTLY where a real-space mask only
// removes it approximately. On a scrambled trial structure the raw map
// correlation reads 0.89 and this reads 0.00.
//
// The observed map is no longer needed by the swarm at all. It is still built,
// for the Harker analysis and the display, but it is not a kernel input.
//
// WHY THE ORBIT MULTIPLICITY, NOT AN EXPANDED LIST
// |F| is constant across a Laue orbit, so the calculated powder line is
// m_h * |F(h)|^2 and one structure factor serves the whole orbit. Evaluating
// the expanded list instead would cost order_p times as much: 8x in Pnma,
// 48x in Ia-3, for an identical answer.
//
// WHY GROUPS
// Pawley's division of intensity between overlapped reflections is a fitting
// artefact, not a measurement. Reflections a powder pattern cannot resolve are
// summed into one group on the host and compared as a group, so the fitness
// only ever tests what was actually observed.
//
// LOAD BALANCE
// One thread owns a whole group and loops its members, so the group sums need
// no atomics - WGSL has no f32 atomicAdd, and working around that with
// bitcast compare-exchange in the hottest loop would cost more than the
// imbalance of a few groups holding two or three reflections instead of one.

// MINIMUM IMAGE
// The o-matrix here is NOT the cell's orthogonalisation matrix and the
// coordinates fed to cartDist() are NOT in the cell's own basis. Both refer to
// a REDUCED basis for the same lattice, computed once on the host by
// sharkoReducedCell().
//
// `d - round(d)` picks the representative inside the box [-1/2,1/2)^3. That box
// is a fundamental domain but not the Wigner-Seitz cell, so in a skewed lattice
// it can return an image that is not the nearest one - by up to 4.6 A in a
// rhombohedral cell at alpha = 60. Since this same routine decides both the
// clash penalty and the distance-window restraint, an over-long distance is not
// a diagnostic error: it is a physically impossible structure that the swarm is
// never charged for, competing against the right answer on correlation alone.
//
// In a reduced basis the box contains the ball of radius safeRadius (half the
// smallest perpendicular width), and rounding is provably exact below it. When
// the cell is thin enough that safeRadius falls under the largest distance
// being tested, the host injects MIN_IMAGE_SHELL = 1 and the 27 neighbouring
// translations are searched as well, which is exact at any distance. The common
// case injects 0 and the loop collapses to a single iteration.

// after
struct WyckoffProj { row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32> }

struct ParticleMeta {
    fit: f32,
    cc: f32,
    stepSize: f32,
    curCC: f32,
    curPen: f32,
    tempOf: f32,
    assign: u32,
    pad: u32,
}

struct GlobalState {
    acceptCount: atomic<u32>,
    pad1: u32,
    pad2: u32,
    pad3: u32,
    particles: array<ParticleMeta>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<f32>;
@group(0) @binding(1) var<storage, read_write> stateBuf: GlobalState;
@group(0) @binding(2) var<storage, read> genPack: array<u32>;
@group(0) @binding(3) var<storage, read> symOps: array<f32>;
@group(0) @binding(4) var<storage, read> reflPack: array<u32>;
@group(0) @binding(5) var<storage, read> groupData: array<f32>;
@group(0) @binding(6) var<storage, read> tables: array<f32>;
@group(0) @binding(7) var<storage, read> siteProj: array<WyckoffProj>;
@group(0) @binding(8) var<uniform> params: Params;

struct Params {
    o0: vec4<f32>, o1: vec4<f32>, o2: vec4<f32>,
    r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>,
    nTot: f32, maxSites: f32, numParticles: f32, nGroupsActive: f32,
    nElem: f32, nBondRules: f32, rMinOff: f32, ruleOff: f32,
    fTabOff: f32, nRefl: f32, penClash: f32, penBond: f32,
    penCoord: f32, penScale: f32, centro: f32, seed: u32,
    generation: u32, is_quench: f32,
};

fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn box_muller(seed1: u32, seed2: u32) -> vec2<f32> {
    let u1 = max(f32(pcg_hash(seed1)) / 4294967295.0, 1e-7); 
    let u2 = f32(pcg_hash(seed2)) / 4294967295.0;
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.28318530718 * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

const WG: u32 = 64u;
const MAX_GEN_ATOMS: u32 = 384u; //__MAX_GEN_ATOMS__
const MIN_IMAGE_SHELL: i32 = 0; //__MIN_IMAGE_SHELL__
const MAX_BOND_RULES: u32 = 8u;
const RULE_STRIDE: u32 = 6u;   // [aType, bType, dmin, dmax, count, mode]
// Total coordination slots across all counted rules. Each counted rule keeps
// the distances to its N nearest partners here, which is what makes the
// coordination penalty continuous. The host refuses a set of rules needing
// more than this.
const MAX_COORD_SLOTS: u32 = 16u;
// Charged, in Angstrom-equivalent, for a required neighbour that does not exist
// anywhere in the cell. Large enough to dominate any real distance miss.
const NO_PARTNER_CHARGE: f32 = 5.0;
// Distance over which the coordination miss goes from linear to quadratic. A
// miss of this size costs twice what a linear hinge would; twice this, three
// times. Small enough that a near-miss is still cheap to explore through.
const COORD_SOFT_A: f32 = 0.25;
// Size of the per-reflection scattering factors held per thread. The type is a
// 4-bit field in genPack so 16 is the hard ceiling, but a cell with more than
// eight distinct elements is not something this program is aimed at, and a
// smaller array is likelier to survive in registers rather than being spilled
// to per-thread scratch - which is what dynamic indexing usually costs. Raise
// it if a structure ever needs to; nLoad below clamps to it either way.
const MAX_ELEM: u32 = 8u;
const TWO_PI: f32 = 6.28318530718;
const NO_NEIGHBOUR: f32 = 1.0e9;

var<workgroup> gx: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gy: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gz: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gT: array<u32, MAX_GEN_ATOMS>;

// Five running sums for the correlation, plus the penalty.
var<workgroup> rIo:   array<f32, WG>;
var<workgroup> rIo2:  array<f32, WG>;
var<workgroup> rIc:   array<f32, WG>;
var<workgroup> rIc2:  array<f32, WG>;
var<workgroup> rIoIc: array<f32, WG>;
var<workgroup> rPen:  array<f32, WG>;
var<workgroup> prop_coords: array<f32, 72>;

fn cartDist(p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let d = p1 - p2;
    // Into the reduced basis. r is integer, so this is exact.
    var q = vec3<f32>(
        params.r0.x*d.x + params.r0.y*d.y + params.r0.z*d.z,
        params.r1.x*d.x + params.r1.y*d.y + params.r1.z*d.z,
        params.r2.x*d.x + params.r2.y*d.y + params.r2.z*d.z);
    q = q - round(q);

    var best: f32 = 1.0e30;
    for (var i = -MIN_IMAGE_SHELL; i <= MIN_IMAGE_SHELL; i = i + 1) {
        for (var j = -MIN_IMAGE_SHELL; j <= MIN_IMAGE_SHELL; j = j + 1) {
            for (var k = -MIN_IMAGE_SHELL; k <= MIN_IMAGE_SHELL; k = k + 1) {
                let t = q + vec3<f32>(f32(i), f32(j), f32(k));
                let cx = params.o0.x*t.x + params.o0.y*t.y + params.o0.z*t.z;
                let cy = params.o1.x*t.x + params.o1.y*t.y + params.o1.z*t.z;
                let cz = params.o2.x*t.x + params.o2.y*t.y + params.o2.z*t.z;
                best = min(best, cx*cx + cy*cy + cz*cz);
            }
        }
    }
    return sqrt(best);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgId: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {

    let pIdx = wgId.x;
    if (pIdx >= u32(params.numParticles)) { return; }   // workgroup-uniform

    let nTot     = min(u32(params.nTot), MAX_GEN_ATOMS);
    let maxSites = u32(params.maxSites);
    let nElem    = u32(params.nElem);
    // nElem is a STRIDE into two tables, so it must not be clamped. This is the
    // separate bound for filling fLoc, whose size is fixed: an out-of-range
    // write there is undefined behaviour rather than an error. The two can only
    // differ if a structure ever carried more than 16 elements, which genPack's
    // 4-bit type field already forbids.
    let nLoad    = min(nElem, MAX_ELEM);
    let nRules   = min(u32(params.nBondRules), MAX_BOND_RULES);
    let rMinOff  = u32(params.rMinOff);
    let ruleOff  = u32(params.ruleOff);
    let nG       = u32(params.nGroupsActive);
    let fTabOff  = u32(params.fTabOff);
    let centro   = params.centro > 0.5;

  let A     = stateBuf.particles[pIdx].assign;
    let gBase = A * nTot;
    let pBase = pIdx * maxSites * 3u;

    // --- PROPOSAL AND PROJECTION PHASE ---
    if (lid == 0u) {
        let mSites = u32(params.maxSites);
        if (params.is_quench > 0.5) {
            for (var s = 0u; s < mSites; s = s + 1u) {
                prop_coords[s*3u]      = particles[pBase + s*3u];
                prop_coords[s*3u + 1u] = particles[pBase + s*3u + 1u];
                prop_coords[s*3u + 2u] = particles[pBase + s*3u + 2u];
            }
        } else {
            let step = stateBuf.particles[pIdx].stepSize;
            let base_seed = params.seed ^ pIdx ^ params.generation;

            for (var s = 0u; s < mSites; s = s + 1u) {
                let cx = particles[pBase + s*3u];
                let cy = particles[pBase + s*3u + 1u];
                let cz = particles[pBase + s*3u + 2u];

                // Hash `s` to guarantee completely independent seeds per coordinate
                let s_hash = pcg_hash(s);
                let bm1 = box_muller(base_seed ^ s_hash, base_seed ^ (s_hash + 1u));
                let bm2 = box_muller(base_seed ^ (s_hash + 2u), base_seed ^ (s_hash + 3u));
                
                var raw_pos = vec3<f32>(cx + bm1.x * step, cy + bm1.y * step, cz + bm2.x * step);
                raw_pos = raw_pos - floor(raw_pos); // CRITICAL: Wrap BEFORE projection
                
                let proj = siteProj[A * mSites + s];
                
                var new_pos = vec3<f32>(
                    dot(proj.row0, vec4<f32>(raw_pos, 1.0)),
                    dot(proj.row1, vec4<f32>(raw_pos, 1.0)),
                    dot(proj.row2, vec4<f32>(raw_pos, 1.0))
                );
                new_pos = new_pos - floor(new_pos);
                
                prop_coords[s*3u]      = new_pos.x;
                prop_coords[s*3u + 1u] = new_pos.y;
                prop_coords[s*3u + 2u] = new_pos.z;
            }
        }
    }
    workgroupBarrier();

    // --- 1. Generate the cell contents ---------------------------------
    // Site coordinates arrive already projected onto their Wyckoff subspace by
    // the host; the operator indices are the position's precomputed coset
    // representatives, so exactly nTot distinct atoms result and no
    // coincidence test is needed.
    for (var g = lid; g < nTot; g = g + WG) {
        let pk = genPack[gBase + g];
        let s  = pk & 0xFFu;
        let o  = (pk >> 8u) & 0xFFFu;
        let ty = (pk >> 20u) & 0xFu;

        let p = vec3<f32>(prop_coords[s * 3u], prop_coords[s * 3u + 1u], prop_coords[s * 3u + 2u]);

        let ob = o * 12u;
        var n = vec3<f32>(
            p.x*symOps[ob+0u] + p.y*symOps[ob+1u] + p.z*symOps[ob+2u] + symOps[ob+9u],
            p.x*symOps[ob+3u] + p.y*symOps[ob+4u] + p.z*symOps[ob+5u] + symOps[ob+10u],
            p.x*symOps[ob+6u] + p.y*symOps[ob+7u] + p.z*symOps[ob+8u] + symOps[ob+11u]
        );
        n = n - floor(n);

        gx[g] = n.x; gy[g] = n.y; gz[g] = n.z;
        gT[g] = ty;
    }
    workgroupBarrier();

    // --- 2. Correlation ------------------------------------------------
    // One thread per group, striding. Structure factors are computed from the
    // generated cell contents, so F(h) is correct as it stands and the orbit
    // is accounted for by the stored multiplicity.
    var sIo: f32 = 0.0; var sIo2: f32 = 0.0;
    var sIc: f32 = 0.0; var sIc2: f32 = 0.0; var sIoIc: f32 = 0.0;

    for (var gi = lid; gi < nG; gi = gi + WG) {
        let gb = gi * 3u;
        let start = u32(groupData[gb]);
        let count = u32(groupData[gb + 1u]);
        let io    = groupData[gb + 2u];

        var ic: f32 = 0.0;
        for (var m = 0u; m < count; m = m + 1u) {
            let rb = (start + m) * 2u;
            let pk = reflPack[rb];
            let h = f32(i32(pk & 0x3FFu) - 512);
            let k = f32(i32((pk >> 10u) & 0x3FFu) - 512);
            let l = f32(i32((pk >> 20u) & 0x3FFu) - 512);
            let mult = f32(reflPack[rb + 1u]);

            // Scattering factor depends on the reflection AND the element, so
            // it is read per (reflection, type) from the precomputed table. The
            // resolution fall-off and the thermal factor live here, on the
            // MODEL - never as a rescaling of the observations, which would put
            // the true structure's correlation below 1.
            // The scattering factor depends on the reflection and the ELEMENT,
            // of which there are nElem - three for PbSO4 - not on the atom, of
            // which there are nTot: twenty-four. Reading it inside the atom loop
            // meant twenty-four scattered global loads per reflection where three
            // would do, in the hottest loop in the program. Hoisted into
            // registers, so the loop below touches nothing but workgroup memory.
            let fb = fTabOff + (start + m) * nElem;
            var fLoc: array<f32, MAX_ELEM>;
            for (var e = 0u; e < nLoad; e = e + 1u) { fLoc[e] = groupData[fb + e]; }

            var re: f32 = 0.0; var im: f32 = 0.0;
            if (centro) {
                // F is real. In a centrosymmetric group the generated contents
                // come in +/- pairs about the origin, so every sine term is
                // cancelled by its partner - the sum is zero by construction,
                // not merely small. Computing it anyway is half the
                // transcendentals in the kernel spent to confirm a zero.
                for (var j = 0u; j < nTot; j = j + 1u) {
                    let ph = TWO_PI * (h * gx[j] + k * gy[j] + l * gz[j]);
                    re = re + fLoc[gT[j]] * cos(ph);
                }
            } else {
                for (var j = 0u; j < nTot; j = j + 1u) {
                    let ph = TWO_PI * (h * gx[j] + k * gy[j] + l * gz[j]);
                    let fj = fLoc[gT[j]];
                    re = re + fj * cos(ph);
                    im = im + fj * sin(ph);
                }
            }
            ic = ic + mult * (re * re + im * im);
        }

        sIo = sIo + io;   sIo2 = sIo2 + io * io;
        sIc = sIc + ic;   sIc2 = sIc2 + ic * ic;
        sIoIc = sIoIc + io * ic;
    }

    // --- 3. Distance constraints ---------------------------------------
    //
    // Three different things, and conflating them is how a constraint ends up
    // meaning something the user did not ask for.
    //
    // A LOWER bound is a true per-pair constraint: no atom of A may ever be
    // closer than dmin to one of B, whatever else is asked. It lives in the
    // rMin matrix and is charged below as a clash, so it applies even when the
    // line asked only for a count.
    //
    // An UPPER bound alone is NOT a per-pair constraint - most Pb-O pairs in a
    // cell are legitimately far apart, and penalising every long one is
    // meaningless. It is a nearest-neighbour condition: every S must have SOME
    // O within 1.65 A, evaluated against the closest partner of that type.
    //
    // A COORDINATION NUMBER is a count of the partners inside [dmin, dmax].
    // Four oxygens around every sulfur is a statement about how many, not about
    // the nearest, and no combination of the other two expresses it.
    var ruleA: array<u32, MAX_BOND_RULES>;
    var ruleB: array<u32, MAX_BOND_RULES>;
    var ruleMin: array<f32, MAX_BOND_RULES>;
    var ruleMax: array<f32, MAX_BOND_RULES>;
    var ruleN: array<f32, MAX_BOND_RULES>;
    var ruleMode: array<u32, MAX_BOND_RULES>;   // 0 none, 1 exactly N, 2 at least N
    for (var k = 0u; k < nRules; k = k + 1u) {
        let rb = ruleOff + k * RULE_STRIDE;
        ruleA[k] = u32(tables[rb]);
        ruleB[k] = u32(tables[rb + 1u]);
        ruleMin[k] = tables[rb + 2u];
        ruleMax[k] = tables[rb + 3u];
        ruleN[k] = tables[rb + 4u];
        ruleMode[k] = u32(tables[rb + 5u]);
    }

    // Where each counted rule's N-nearest window starts. A prefix sum, so the
    // slots pack tightly and the host only has to check the total.
    var slotOff: array<u32, MAX_BOND_RULES>;
    {
        var acc: u32 = 0u;
        for (var k = 0u; k < nRules; k = k + 1u) {
            slotOff[k] = acc;
            if (ruleMode[k] != 0u) { acc = acc + u32(ruleN[k]); }
        }
    }

    var pen: f32 = 0.0;
    for (var i = lid; i < nTot; i = i + WG) {
        let ti = gT[i];
        let pi = vec3<f32>(gx[i], gy[i], gz[i]);
        let rowMin = rMinOff + ti * nElem;

        var myRules: u32 = 0u;
        for (var k = 0u; k < nRules; k = k + 1u) {
            if (ruleA[k] == ti) { myRules = myRules | (1u << k); }
        }
        var nearest: array<f32, MAX_BOND_RULES>;
        var inWindow: array<f32, MAX_BOND_RULES>;
        for (var k = 0u; k < MAX_BOND_RULES; k = k + 1u) {
            nearest[k] = NO_NEIGHBOUR;
            inWindow[k] = 0.0;
        }
        // The N nearest partners of each counted rule, ascending. Kept as
        // DISTANCES, not as a count - see the penalty block below.
        var slot: array<f32, MAX_COORD_SLOTS>;
        for (var m = 0u; m < MAX_COORD_SLOTS; m = m + 1u) { slot[m] = NO_NEIGHBOUR; }

        for (var j = 0u; j < nTot; j = j + 1u) {
            if (j == i) { continue; }
            let tj = gT[j];
            let d = cartDist(pi, vec3<f32>(gx[j], gy[j], gz[j]));

            // Charged once per unordered pair. This also covers an atom
            // against its own symmetry mates, which with exact cosets are
            // genuinely distinct atoms - and it self-corrects the degenerate
            // case where a free parameter drifts onto a higher-symmetry
            // sub-position, since the images that ought to have merged
            // coincide instead and the clash pushes the parameter back off.
            let dmin = tables[rowMin + tj];
            if (j > i && d < dmin) {
                // Scaled by how deep the overlap runs. A flat charge made a
                // 0.3 A lead-oxygen contact cost exactly what a 2.1 A one did,
                // so the swarm had no gradient telling it which way out - and
                // structures with atoms essentially on top of each other
                // survived to the end of the run.
                let overlap = (dmin - d) / max(dmin, 1e-3);
                pen = pen + params.penClash * (1.0 + 9.0 * overlap);
            }

            if (myRules != 0u) {
                for (var k = 0u; k < nRules; k = k + 1u) {
                    if ((myRules & (1u << k)) == 0u) { continue; }
                    if (ruleB[k] != tj) { continue; }
                    nearest[k] = min(nearest[k], d);
                    // The counting window is closed at both ends. dmax is +inf
                    // when the constraint left the upper side open, which
                    // compares correctly without a sentinel.
                    if (d >= ruleMin[k] && d <= ruleMax[k]) {
                        inWindow[k] = inWindow[k] + 1.0;
                    }
                    // Insertion into the rule's sorted N-nearest window.
                    let nk = u32(ruleN[k]);
                    if (ruleMode[k] != 0u && nk > 0u) {
                        let lo = slotOff[k];
                        let hi = lo + nk;              // exclusive
                        if (d < slot[hi - 1u]) {
                            var m = hi - 1u;
                            loop {
                                if (m > lo && slot[m - 1u] > d) {
                                    slot[m] = slot[m - 1u];
                                    m = m - 1u;
                                } else { break; }
                            }
                            slot[m] = d;
                        }
                    }
                }
            }
        }

        for (var k = 0u; k < nRules; k = k + 1u) {
            if ((myRules & (1u << k)) == 0u) { continue; }

            if (ruleMode[k] == 0u) {
                // Nearest-neighbour condition.
                let nk = nearest[k];
                // An atom with no partner of the required type anywhere is a
                // badly posed constraint, not a bad structure; skip rather than
                // charge it.
                if (nk < NO_NEIGHBOUR && nk > ruleMax[k]) {
                    pen = pen + params.penBond * (nk - ruleMax[k]);
                }
            } else {
                // Coordination number, charged BY DISTANCE rather than by count.
                //
                // Counting is the obvious implementation and it does not work.
                // The number of O within 1.4-1.9 A of an S is an integer that
                // changes only when an atom crosses a window edge, so the
                // penalty is a step function: on PbSO4 it sat at its maximum
                // across 100% of the search space, a constant offset subtracted
                // from every particle equally. It cost the swarm nothing to
                // ignore and told it nothing about which way to move, and the
                // run converged on structures with no S-O bond at all while
                // reporting the constraint as maximally violated.
                //
                // What is needed is a force, so the charge is the distance the N
                // NEAREST partners lie outside the window. That is continuous:
                // as S drifts towards an O the charge falls smoothly, and the
                // swarm has a gradient to descend all the way in from wherever
                // it started.
                let nk = u32(ruleN[k]);
                var deficit: f32 = 0.0;
                for (var m = 0u; m < nk; m = m + 1u) {
                    let dm = slot[slotOff[k] + m];
                    if (dm >= NO_NEIGHBOUR) {
                        // Fewer atoms of that type exist than the rule demands.
                        deficit = deficit + NO_PARTNER_CHARGE;
                    } else {
                        // The miss grows QUADRATICALLY, not linearly.
                        //
                        // A plain hinge has a constant restoring force - 0.02 CC
                        // units per Angstrom, the whole way in - so any opposing
                        // force larger than that parks the bond just outside the
                        // window and holds it there. On PbSO4 that is exactly
                        // what happened: two different Wyckoff assignments both
                        // converged to S-O = 2.00 A, 0.094 A outside a window
                        // ending at 1.90, with the four distances equal to
                        // within 0.002 A. The search was not failing; it had
                        // found the balance point of the function it was given.
                        //
                        // Squaring the miss makes the force grow with the
                        // violation, so a fixed opposing force cannot hold the
                        // bond out indefinitely - there is always a distance
                        // beyond which the constraint wins.
                        let miss = max(0.0, dm - ruleMax[k]) + max(0.0, ruleMin[k] - dm);
                        deficit = deficit + miss * (1.0 + miss / COORD_SOFT_A);
                    }
                }
                pen = pen + params.penBond * deficit;

                // Surplus is still a count: an atom that has SIX partners in a
                // window asking for four is wrong by two, and no distance
                // expresses that. Only "exactly N" charges it - "at least N" is
                // satisfied by any surplus.
                if (ruleMode[k] == 1u) {
                    pen = pen + params.penCoord * max(0.0, inWindow[k] - ruleN[k]);
                }
            }
        }
    }

    // The ramp. Penalties start soft and end decisive.
    //
    // A hard rejection cannot work here: in a tight cell most RANDOM initial
    // particles clash, so every one of them would score the same -inf and the
    // swarm would have no gradient anywhere to move along. The penalty is what
    // tells a particle which way is out.
    //
    // But a penalty that stays soft lets a physically impossible structure
    // survive on correlation alone. Scaling it up over the run gives both: the
    // early generations explore freely, and by the end a clash costs more than
    // any correlation can repay. The resolution ramp above works the same way
    // and for the same reason.
    pen = pen * params.penScale;

    // --- 4. Reduce ------------------------------------------------------
    rIo[lid] = sIo; rIo2[lid] = sIo2; rIc[lid] = sIc;
    rIc2[lid] = sIc2; rIoIc[lid] = sIoIc; rPen[lid] = pen;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            rIo[lid]   = rIo[lid]   + rIo[lid + stride];
            rIo2[lid]  = rIo2[lid]  + rIo2[lid + stride];
            rIc[lid]   = rIc[lid]   + rIc[lid + stride];
            rIc2[lid]  = rIc2[lid]  + rIc2[lid + stride];
            rIoIc[lid] = rIoIc[lid] + rIoIc[lid + stride];
            rPen[lid]  = rPen[lid]  + rPen[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let n = f32(nG);
        let num = n * rIoIc[0] - rIo[0] * rIc[0];
        let va = max(0.0, n * rIo2[0] - rIo[0] * rIo[0]);
        let vb = max(0.0, n * rIc2[0] - rIc[0] * rIc[0]);
        let den = sqrt(va * vb);
        // A trial structure whose calculated intensities are all equal has zero
        // variance and no correlation is defined. That happens for an empty or
        // fully collapsed cell, which deserves the worst score, not a NaN that
        // would poison every personal best it touched.
        var cc: f32 = -1.0;
        if (den > 1e-12) { cc = num / den; }

        // Two numbers, not one. The swarm must optimise the PENALISED score, or
        // the restraints do nothing; but the ranking a user reads should be the
        // map correlation itself, and a candidate carrying a large penalty is a
        // different statement from one that simply fits badly. Writing only
        // `cc - penalty` and calling the result "CC" conflates them: a structure
        // with a correlation of 0.98 and one clash would be reported as 0.93 and
        // ranked below an unrestrained 0.95 with no way to see why.
        //
        // Slot [pIdx] is the score the search follows; [numParticles + pIdx] is
        // the bare correlation.
        stateBuf.particles[pIdx].fit = cc - rPen[0];
        stateBuf.particles[pIdx].cc = cc;

        // --- METROPOLIS ACCEPT/REJECT PHASE ---
        if (params.is_quench < 0.5) {
            let fProp = cc - rPen[0];
            let cCC = stateBuf.particles[pIdx].curCC;
            let cPen = stateBuf.particles[pIdx].curPen;
            
            var fCur: f32 = -1e30;
            if (cCC > -2.0) { fCur = cCC - cPen * params.penScale; }

            var take = false;
            if (fProp >= fCur || fCur < -9999.0) {
                take = true;
            } else {
                let threshold = exp((fProp - fCur) / stateBuf.particles[pIdx].tempOf);
                let rand_val = f32(pcg_hash(params.seed ^ pIdx ^ 9999u)) / 4294967295.0;
                if (rand_val < threshold) { take = true; }
            }

let step = stateBuf.particles[pIdx].stepSize;
            if (take) {
                let mSites = u32(params.maxSites);
                for (var s = 0u; s < mSites; s = s + 1u) {
                    particles[pBase + s*3u]      = prop_coords[s*3u];
                    particles[pBase + s*3u + 1u] = prop_coords[s*3u + 1u];
                    particles[pBase + s*3u + 2u] = prop_coords[s*3u + 2u];
                }
                stateBuf.particles[pIdx].curCC  = cc;
                stateBuf.particles[pIdx].curPen = rPen[0] / params.penScale; 
                stateBuf.particles[pIdx].stepSize = min(0.5, step * 1.058);
                atomicAdd(&stateBuf.acceptCount, 1u);
            } else {
                stateBuf.particles[pIdx].stepSize = max(0.0002, step * 0.976);
            }
        }
    }
}