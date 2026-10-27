const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const context = vm.createContext({ console: { log() {}, warn() {}, error() {} },
    document: { getElementById() { return null; } }, state: { applyLP: true } });
const start = html.indexOf('        function parseReflectionHeader(');
const end = html.indexOf('        /** Human-readable description of the intensity route');
vm.runInContext(html.slice(start, end), context);
vm.runInContext(fs.readFileSync(path.join(root, 'symmetry_utils.js'), 'utf8'), context);
const parse = text => context.parsePeakFile(text);
const sample = fs.readFileSync(path.join(root, 'test_files/wppf_results_202610270622.txt'), 'utf8');
const data = parse(sample);
assert.equal(data.reflections.length, 392);
assert.equal(data.spaceGroup.number, 12);
assert.equal(data.spaceGroup.name, 'I12/m1');
assert.equal(data.cell.beta, 93.06513);
assert.equal(data.cell.alpha, 90);
assert.equal(data.wavelength, 0.67776);
assert.equal(data.polarisationK, 1);
assert.equal(data.reflections[0].Fo, 0);
assert.equal(data.reflections[1].l, -1);
assert.ok(data.parseWarnings.some(w => w.includes('overlapping')));
context.applyIntensityCorrections(data);
assert.equal(data.perReflection, true);
assert.ok(Math.abs(data.reflections[1].intensity - 7.723530e-4) < 1e-15);
assert.ok(data.reflections.every(r => Number.isFinite(r.intensity)));
assert.throws(() => parse(sample.replace('7.723530E-4', 'unavailable')), /Invalid WPPF reflection/);
assert.throws(() => parse(sample + '\n### Phase other (p2)'), /one dataset and one phase/);
for (const name of fs.readdirSync(path.join(root, 'test_files')).filter(n => /Pawley.*\.txt$/i.test(n))) {
    const legacy = parse(fs.readFileSync(path.join(root, 'test_files', name), 'utf8'));
    context.applyIntensityCorrections(legacy);
    assert.ok(legacy.reflections.length > 0);
    assert.ok(legacy.reflections.every(r => Number.isFinite(r.intensity)));
    console.log(`${name}: ${legacy.reflections.length} reflections`);
}
assert.equal(context.parseReflectionHeader('h k l 2theta I_hkl').join(','), 'h,k,l,2theta,I_hkl');
for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)) new vm.Script(match[1]);
for (const name of fs.readdirSync(root).filter(n => n.endsWith('.js'))) {
    new vm.Script(fs.readFileSync(path.join(root, name), 'utf8'), { filename: name });
}
console.log('WPPF import, legacy imports, and JavaScript syntax checks passed.');
