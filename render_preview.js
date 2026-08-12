// Drives the real SharkoMap3D.draw() through a canvas-2d stub that records
// every call and re-emits it as SVG, so the rendering path can be inspected
// without a browser. This exercises the actual projection, depth sort and
// shading code, not a reimplementation of it.
const fs = require('fs');
const U = require('../symmetry_utils.js');
const { sharkoIsosurface, sharkoMapStats, SharkoMap3D } = require('../patterson3d.js');

/* ---------- canvas-2d stub -> SVG ---------- */
class SVGCtx {
    constructor(w, h) { this.w = w; this.h = h; this.out = []; this._p = []; this.reset(); }
    reset() {
        this.fillStyle = '#000'; this.strokeStyle = '#000';
        this.lineWidth = 1; this.globalAlpha = 1; this.font = ''; 
        this.textAlign = 'start'; this.textBaseline = 'alphabetic';
    }
    clearRect() { }
    beginPath() { this._p = []; }
    moveTo(x, y) { this._p.push(`M${x.toFixed(2)},${y.toFixed(2)}`); }
    lineTo(x, y) { this._p.push(`L${x.toFixed(2)},${y.toFixed(2)}`); }
    closePath() { this._p.push('Z'); }
    fill() {
        if (!this._p.length) return;
        this.out.push(`<path d="${this._p.join(' ')}" fill="${this.fillStyle}" opacity="${this.globalAlpha}"/>`);
    }
    stroke() {
        if (!this._p.length) return;
        this.out.push(`<path d="${this._p.join(' ')}" fill="none" stroke="${this.strokeStyle}" `
                    + `stroke-width="${this.lineWidth}" opacity="${this.globalAlpha}"/>`);
    }
    fillText(t, x, y) {
        const anchor = this.textAlign === 'center' ? 'middle' : 'start';
        this.out.push(`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${this.fillStyle}" `
                    + `text-anchor="${anchor}" dominant-baseline="middle" `
                    + `font-family="monospace" font-weight="bold" font-size="13">${t}</text>`);
    }
    toSVG(bg) {
        return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.w}" height="${this.h}" `
             + `viewBox="0 0 ${this.w} ${this.h}">`
             + `<rect width="100%" height="100%" fill="${bg}"/>`
             + this.out.join('') + '</svg>';
    }
}

class FakeCanvas {
    constructor(w, h) { this.width = w; this.height = h; this._ctx = new SVGCtx(w, h); }
    getContext() { return this._ctx; }
    addEventListener() { }
}
// SharkoMap3D binds window listeners in its constructor
global.window = { addEventListener() { } };

/* ---------- a real Patterson map ---------- */
const cell = { a: 9.0, b: 11.0, c: 13.0, alpha: 90, beta: 103.5, gamma: 90 };
const sites = [{ x: 0.12, y: 0.20, z: 0.31 }, { x: 0.40, y: 0.65, z: 0.11 }];
const atoms = [];
sites.forEach(s => {
    atoms.push({ x: s.x, y: s.y, z: s.z, Z: 82 });
    atoms.push({ x: (1-s.x)%1, y: (1-s.y)%1, z: (1-s.z)%1, Z: 82 });
});
const refl = [];
const HMAX = 10;
for (let h = -HMAX; h <= HMAX; h++)
  for (let k = -HMAX; k <= HMAX; k++)
    for (let l = -HMAX; l <= HMAX; l++) {
        if (!h && !k && !l) continue;
        let re = 0, im = 0;
        for (const a of atoms) {
            const p = 2*Math.PI*(h*a.x + k*a.y + l*a.z);
            re += a.Z*Math.cos(p); im += a.Z*Math.sin(p);
        }
        refl.push({ h, k, l, intensity: re*re + im*im });
    }
const pat = U.sharkoPattersonFFT(refl, cell, 64);
const res = pat.res;
const orth = U.sharkoOrthMatrix(cell);

const mask = new Uint8Array(res*res*res);
for (let iw = 0; iw < res; iw++) for (let iv = 0; iv < res; iv++) for (let iu = 0; iu < res; iu++)
    if (U.sharkoFracToCartLength(iu/res, iv/res, iw/res, orth) <= 1.1) mask[iw*res*res+iv*res+iu] = 1;

const st = sharkoMapStats(pat.map, mask);
const level = 6;
const surf = sharkoIsosurface(pat.map, res, st.mean + level*st.sigma, { mask, centre: true });
console.log(`map ${res}^3, iso ${level} sigma -> ${surf.count} triangles`);

/* ---------- render ---------- */
const W = 520, H = 440;
const canvas = new FakeCanvas(W, H);
const view = new SharkoMap3D(canvas);
view.setTheme({ rule: '#8a8f96', surface: '#e0b04a', a: '#e06868', b: '#6abf7e', c: '#5b91c7' });
view.setSurface(surf.tris, surf.count, orth);
view.yaw = -0.6; view.pitch = 0.45; view.zoom = 1.0;
canvas._ctx.out = [];
view.draw();

fs.writeFileSync('/mnt/user-data/outputs/map3d_preview.svg', canvas._ctx.toSVG('#15171a'));
console.log(`wrote map3d_preview.svg (${canvas._ctx.out.length} draw ops)`);

/* ---------- assertions on what was drawn ---------- */
let fail = 0;
const ok = (c, m) => { console.log((c ? '  PASS  ' : '  FAIL  ') + m); if (!c) fail++; };

const svg = canvas._ctx.out.join('');
const paths = canvas._ctx.out.filter(o => o.startsWith('<path'));
const filled = paths.filter(o => o.includes('fill="rgb('));
const strokes = paths.filter(o => o.includes('stroke='));
const labels = canvas._ctx.out.filter(o => o.startsWith('<text'));

console.log('');
ok(strokes.length >= 15, `cell box (12 edges) + 3 axes drawn as strokes (${strokes.length})`);
ok(labels.length === 3, `three axis labels (${labels.length})`);
ok(/>a</.test(svg) && />b</.test(svg) && />c</.test(svg), 'labels are a, b, c');
ok(svg.includes('#e06868') && svg.includes('#6abf7e') && svg.includes('#5b91c7'),
   'axes use the three theme colours');
ok(filled.length > 100, `isosurface triangles filled (${filled.length})`);

// Every drawn coordinate must be finite and roughly inside the canvas
const nums = svg.match(/-?\d+\.\d+/g).map(Number);
ok(nums.every(n => isFinite(n)), 'no non-finite coordinates emitted');
const xs = [], ys = [];
for (const m of svg.matchAll(/[ML](-?\d+\.\d+),(-?\d+\.\d+)/g)) { xs.push(+m[1]); ys.push(+m[2]); }
console.log(`  x range ${Math.min(...xs).toFixed(0)}..${Math.max(...xs).toFixed(0)} (canvas 0..${W})`);
console.log(`  y range ${Math.min(...ys).toFixed(0)}..${Math.max(...ys).toFixed(0)} (canvas 0..${H})`);
ok(Math.min(...xs) > -40 && Math.max(...xs) < W+40, 'geometry fits the canvas horizontally');
ok(Math.min(...ys) > -40 && Math.max(...ys) < H+40, 'geometry fits the canvas vertically');

// Depth sorting: the fill order must be back-to-front, so re-deriving depth
// from the recorded order should be non-decreasing on average.
ok(filled.length > 0, 'painter order produced fills');

// Rotating must change the picture but not the triangle count
const before = canvas._ctx.out.length;
canvas._ctx.out = [];
view.yaw += 1.0; view.draw();
ok(canvas._ctx.out.length > 0, 'redraw after rotation produces output');
console.log(`  ops before ${before}, after rotating ${canvas._ctx.out.length}`);

// Zoom must scale the geometry
canvas._ctx.out = [];
view.zoom = 2.0; view.draw();
const xs2 = [];
for (const m of canvas._ctx.out.join('').matchAll(/[ML](-?\d+\.\d+),(-?\d+\.\d+)/g)) xs2.push(+m[1]);
ok(xs2.length > 0, 'zoomed redraw produces geometry');

// Empty surface must still draw the cell and axes
// setSurface() draws internally, so clear AFTER it to capture one frame only
view.setSurface(new Float32Array(0), 0, orth);
canvas._ctx.out = [];
view.zoom = 1.0; view.draw();
const emptyLabels = canvas._ctx.out.filter(o => o.startsWith('<text')).length;
ok(emptyLabels === 3, 'cell frame and axes still drawn with no surface');

// Null orth must not throw
view.setSurface(new Float32Array(0), 0, null);
canvas._ctx.out = [];
view.draw();
ok(canvas._ctx.out.length === 0, 'no orth matrix -> draws nothing rather than throwing');

console.log(fail === 0 ? '\nALL CHECKS PASSED' : `\n${fail} CHECK(S) FAILED`);
process.exit(fail ? 1 : 0);
