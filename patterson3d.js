/**
 * 3D structure view and volume export for sHarko.
 *
 * The marching-tetrahedra isosurface extractor that used to live here has been
 * removed along with the 3D view of the Patterson map itself: a map isosurface
 * showed nothing the 2D sections did not show more legibly, and the blobs were
 * hard to read at any level. What remains is the structure viewer - which
 * displays atoms, not density - plus the map statistics the colour scale needs
 * and the .grd writer for anyone who wants to inspect the volume in VESTA.
 */

/**
 * Mean and standard deviation of a map, ignoring masked voxels.
 *
 * The isolevel is expressed in standard deviations above the mean rather than
 * in raw map units, because raw units depend entirely on how the input
 * intensities happened to be scaled - "3 sigma" means the same thing whatever
 * the file contained, and is how crystallographers describe map features.
 */
function sharkoMapStats(map, mask) {
    let n = 0, sum = 0, sumSq = 0;
    for (let i = 0; i < map.length; i++) {
        if (mask && mask[i]) continue;
        const v = map[i];
        if (!isFinite(v)) continue;
        n++; sum += v; sumSq += v * v;
    }
    if (!n) return { mean: 0, sigma: 0, count: 0 };
    const mean = sum / n;
    const varr = Math.max(0, sumSq / n - mean * mean);
    return { mean, sigma: Math.sqrt(varr), count: n };
}

/* ------------------------------------------------------------------ */
/*  Renderer (three.js)                                                */
/* ------------------------------------------------------------------ */

/**
 * WebGL isosurface view built on three.js, which must already be loaded
 * (lib/three.min.js) before this file runs.
 *
 * This replaces a painter's-algorithm renderer that sorted and filled every
 * triangle on a 2D canvas each frame. That was fine for the few thousand
 * triangles a high isolevel produces, but a real Patterson map at the default
 * 3 sigma yields around 20 000, and at 1.5 sigma nearly 60 000 - tens of
 * thousands of canvas fill calls per frame, which made rotation stutter.
 * Handing the mesh to the GPU once and re-rendering only the camera makes the
 * triangle count almost irrelevant.
 *
 * Orbit control is implemented directly rather than pulling in OrbitControls:
 * it is thirty lines, and it avoids depending on a second file whose path and
 * module format vary between three.js distributions.
 */
class SharkoMap3D {
    constructor(canvas) {
        this.canvas = canvas;
        this.ok = false;
        if (typeof THREE === 'undefined') {
            this._fail('three.js not loaded - expected lib/three.min.js');
            return;
        }
        try {
            this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        } catch (e) {
            this._fail('WebGL unavailable: ' + e.message);
            return;
        }

        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);

        // Two lights plus ambient so the surface reads as solid from any angle.
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
        const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(1, 1.4, 1);
        const fill = new THREE.DirectionalLight(0xffffff, 0.30); fill.position.set(-1, -0.6, -0.8);
        this.scene.add(key, fill);

        this.root = new THREE.Group();
        this.scene.add(this.root);

        this.surfaceMesh = null;
        this.boxGroup = null;
        this.axesGroup = null;
        this.markerGroup = null;

        this.orth = null;
        this.radius = 1;
        this.yaw = -0.6; this.pitch = 0.45; this.zoom = 1.0;
        this.showBox = true;

        this.theme = { rule: 0x8a8f96, surface: 0xe0b04a,
                       a: 0xe06868, b: 0x6abf7e, c: 0x5b91c7, marker: 0xffffff };
        this.ok = true;
        this._bindInput();
    }

    _fail(msg) {
        this.error = msg;
        console.error('[SharkoMap3D]', msg);
    }

    /* ---- theme ---- */
    setTheme(t) {
        const hex = v => {
            if (typeof v === 'number') return v;
            const m = /^#?([a-f\d]{6})$/i.exec(String(v).trim());
            return m ? parseInt(m[1], 16) : 0xffffff;
        };
        for (const k of Object.keys(this.theme)) if (t[k] !== undefined) this.theme[k] = hex(t[k]);
        if (!this.ok) return;
        if (this.surfaceMesh) this.surfaceMesh.material.color.setHex(this.theme.surface);
        this._buildFrame();
        this.draw();
    }

    /* ---- geometry ---- */

    /**
     * Uploads a new isosurface. `tris` is 9 floats per triangle in FRACTIONAL
     * coordinates already centred on the origin; the orth matrix converts to
     * Cartesian so the cell shape is honoured.
     */
    setSurface(tris, count, orthMat) {
        if (!this.ok) return;
        this.orth = orthMat;
        this._computeRadius();

        if (this.surfaceMesh) {
            this.surfaceMesh.geometry.dispose();
            this.root.remove(this.surfaceMesh);
            this.surfaceMesh = null;
        }

        if (count > 0 && orthMat) {
            const pos = new Float32Array(count * 9);
            for (let i = 0; i < count * 3; i++) {
                const b = i * 3;
                const fx = tris[b], fy = tris[b + 1], fz = tris[b + 2];
                pos[b]     = orthMat[0]*fx + orthMat[1]*fy + orthMat[2]*fz;
                pos[b + 1] = orthMat[3]*fx + orthMat[4]*fy + orthMat[5]*fz;
                pos[b + 2] = orthMat[6]*fx + orthMat[7]*fy + orthMat[8]*fz;
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            // Marching tetrahedra does not guarantee consistent winding, so the
            // material is double-sided and normals are computed from the faces.
            geo.computeVertexNormals();
            const mat = new THREE.MeshLambertMaterial({
                color: this.theme.surface, side: THREE.DoubleSide, flatShading: true
            });
            this.surfaceMesh = new THREE.Mesh(geo, mat);
            this.root.add(this.surfaceMesh);
        }

        this._buildFrame();
        this.draw();
    }

    /** Small spheres at given fractional positions (already centred). */
    setMarkers(points) {
        if (!this.ok) return;
        if (this.markerGroup) {
            this.markerGroup.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
            this.root.remove(this.markerGroup);
            this.markerGroup = null;
        }
        if (!points || !points.length || !this.orth) { this.draw(); return; }

        const M = this.orth;
        const g = new THREE.Group();
        const r = this.radius * 0.022;
        const sphere = new THREE.SphereGeometry(r, 12, 10);
        const byColour = new Map();
        for (const p of points) {
            const col = (p.color !== undefined) ? p.color : this.theme.marker;
            if (!byColour.has(col)) {
                byColour.set(col, new THREE.MeshLambertMaterial({ color: col }));
            }
            const m = new THREE.Mesh(sphere, byColour.get(col));
            m.position.set(
                M[0]*p.x + M[1]*p.y + M[2]*p.z,
                M[3]*p.x + M[4]*p.y + M[5]*p.z,
                M[6]*p.x + M[7]*p.y + M[8]*p.z
            );
            if (p.scale) m.scale.setScalar(p.scale);
            g.add(m);
        }
        this.markerGroup = g;
        this.root.add(g);
        this.draw();
    }

    setShowBox(on) { this.showBox = !!on; this._buildFrame(); this.draw(); }

    _computeRadius() {
        this.radius = 1;
        if (!this.orth) return;
        const M = this.orth;
        let max = 1e-6;
        for (let i = 0; i < 8; i++) {
            const fx = (i & 1 ? .5 : -.5), fy = (i & 2 ? .5 : -.5), fz = (i & 4 ? .5 : -.5);
            const X = M[0]*fx + M[1]*fy + M[2]*fz;
            const Y = M[3]*fx + M[4]*fy + M[5]*fz;
            const Z = M[6]*fx + M[7]*fy + M[8]*fz;
            max = Math.max(max, Math.hypot(X, Y, Z));
        }
        this.radius = max;
    }

    /** Unit-cell wireframe and the a/b/c axes from the origin at the centre. */
    _buildFrame() {
        if (!this.ok) return;
        for (const grp of ['boxGroup', 'axesGroup']) {
            if (this[grp]) {
                this[grp].traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
                this.root.remove(this[grp]);
                this[grp] = null;
            }
        }
        if (!this.orth) return;
        const M = this.orth;
        const toCart = (fx, fy, fz) => new THREE.Vector3(
            M[0]*fx + M[1]*fy + M[2]*fz,
            M[3]*fx + M[4]*fy + M[5]*fz,
            M[6]*fx + M[7]*fy + M[8]*fz
        );

        if (this.showBox) {
            const corners = [];
            for (let i = 0; i < 8; i++)
                corners.push(toCart(i & 1 ? .5 : -.5, i & 2 ? .5 : -.5, i & 4 ? .5 : -.5));
            const edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]];
            const pts = [];
            edges.forEach(([p, q]) => { pts.push(corners[p], corners[q]); });
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({ color: this.theme.rule, transparent: true, opacity: 0.7 });
            this.boxGroup = new THREE.LineSegments(geo, mat);
            this.root.add(this.boxGroup);
        }

        const axes = new THREE.Group();
        const O = new THREE.Vector3(0, 0, 0);
        [[[.5,0,0], this.theme.a], [[0,.5,0], this.theme.b], [[0,0,.5], this.theme.c]]
        .forEach(([f, col]) => {
            const geo = new THREE.BufferGeometry().setFromPoints([O, toCart(f[0], f[1], f[2])]);
            axes.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: col })));
        });
        this.axesGroup = axes;
        this.root.add(axes);
    }

    /* ---- camera / input ---- */

    _bindInput() {
        const c = this.canvas;
        let dragging = false, lastX = 0, lastY = 0;
        const pos = e => e.touches ? [e.touches[0].clientX, e.touches[0].clientY] : [e.clientX, e.clientY];
        const down = e => { dragging = true; [lastX, lastY] = pos(e); e.preventDefault(); };
        const move = e => {
            if (!dragging) return;
            const [x, y] = pos(e);
            this.yaw   += (x - lastX) * 0.01;
            this.pitch += (y - lastY) * 0.01;
            const lim = Math.PI / 2 - 0.01;
            this.pitch = Math.max(-lim, Math.min(lim, this.pitch));
            lastX = x; lastY = y;
            this.draw();
            e.preventDefault();
        };
        const up = () => { dragging = false; };
        c.addEventListener('mousedown', down);
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        c.addEventListener('touchstart', down, { passive: false });
        c.addEventListener('touchmove', move, { passive: false });
        c.addEventListener('touchend', up);
        c.addEventListener('wheel', e => {
            this.zoom *= (e.deltaY > 0 ? 0.92 : 1.08);
            this.zoom = Math.max(0.25, Math.min(6, this.zoom));
            this.draw();
            e.preventDefault();
        }, { passive: false });
    }

    /* resetView() lived here. Nothing called it: the orbit state is cheap to
       restore by dragging, and the button that used to invoke it has been
       removed from the interface. */

    setSize(w, h) {
        if (!this.ok || w <= 0 || h <= 0) return;
        this.renderer.setSize(w, h, false);
        this.draw();
    }

    draw() {
        if (!this.ok) return;
        const w = this.canvas.clientWidth || this.canvas.width;
        const h = this.canvas.clientHeight || this.canvas.height;
        if (!w || !h) return;

        // Orthographic frustum sized to the cell so it always fits, whatever
        // the aspect ratio or cell shape.
        const R = (this.radius || 1) * 1.15 / (this.zoom || 1);
        const aspect = w / h;
        this.camera.left = -R * aspect; this.camera.right = R * aspect;
        this.camera.top = R; this.camera.bottom = -R;
        this.camera.near = -10 * R; this.camera.far = 10 * R;
        this.camera.updateProjectionMatrix();

        const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        this.camera.position.set(R * 4 * cp * sy, R * 4 * sp, R * 4 * cp * cy);
        this.camera.up.set(0, 1, 0);
        this.camera.lookAt(0, 0, 0);

        this.renderer.render(this.scene, this.camera);
    }

    /** Data URL of the current frame, for the PDF report. */
    toDataURL() {
        if (!this.ok) return null;
        this.draw();                       // the buffer may have been cleared since the last frame
        try { return this.renderer.domElement.toDataURL('image/png'); }
        catch (e) { return null; }
    }
}

/* ------------------------------------------------------------------ */
/*  Volume export                                                      */
/* ------------------------------------------------------------------ */

/**
 * Writes a map as a VESTA-readable .grd volume file.
 *
 *   line 1  title
 *   line 2  a b c alpha beta gamma
 *   line 3  Nx Ny Nz
 *   rest    values, x fastest then y then z
 *
 * The grid is written with the redundant closing plane included (Nx = res+1),
 * because .grd readers treat the last point along each axis as coincident with
 * the first rather than inferring periodicity. Omitting it puts a visible seam
 * across one face of the cell.
 */
function sharkoBuildGRD(map, res, cell, title) {
    if (!map || !res || !cell) throw new Error('No map to export.');
    const n = res + 1;
    const out = [];
    out.push((title || 'sHarko Patterson map').replace(/[\r\n]+/g, ' '));
    out.push([cell.a, cell.b, cell.c, cell.alpha ?? 90, cell.beta ?? 90, cell.gamma ?? 90]
             .map(v => Number(v).toFixed(6)).join(' '));
    out.push(`${n} ${n} ${n}`);

    const row = [];
    for (let iz = 0; iz < n; iz++) {
        const z = (iz % res) * res * res;
        for (let iy = 0; iy < n; iy++) {
            const y = (iy % res) * res;
            for (let ix = 0; ix < n; ix++) {
                const v = map[z + y + (ix % res)];
                row.push(Number.isFinite(v) ? v.toPrecision(7) : '0');
                if (row.length === 6) { out.push(row.join(' ')); row.length = 0; }
            }
        }
    }
    if (row.length) out.push(row.join(' '));
    return out.join('\n') + '\n';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { sharkoMapStats, SharkoMap3D, sharkoBuildGRD };
}
