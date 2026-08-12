

(function (root) {
    'use strict';

    if (!root.SG_DATABASE) {
        throw new Error('SG_ENGINE: SG_DATABASE must be loaded before sg_engine.js');
    }
    const DB = root.SG_DATABASE.space_groups;

    
    // 1. Point-group -> Laue-class map (the 11 centrosymmetric Laue groups).
    //    Source: International Tables Vol. A, Table 3.2.1.4.
    
    const POINT_GROUP_TO_LAUE = {
        '1':    '-1',   '-1':   '-1',
        '2':    '2/m',  'm':    '2/m',  '2/m':  '2/m',
        '222':  'mmm',  'mm2':  'mmm',  'mmm':  'mmm',
        '4':    '4/m',  '-4':   '4/m',  '4/m':  '4/m',
        '422':  '4/mmm','4mm':  '4/mmm','-42m': '4/mmm','4/mmm':'4/mmm',
        '3':    '-3',   '-3':   '-3',
        '32':   '-3m',  '3m':   '-3m',  '-3m':  '-3m',
        '6':    '6/m',  '-6':   '6/m',  '6/m':  '6/m',
        '622':  '6/mmm','6mm':  '6/mmm','-62m': '6/mmm','6/mmm':'6/mmm',
        '23':   'm-3',  'm-3':  'm-3',
        '432':  'm-3m', '-43m': 'm-3m', 'm-3m': 'm-3m'
    };

    // The 'system' in the main thread's old code used 'rhombohedral' as a
    // distinct bucket; the cctbx JSON keeps them all under 'trigonal'. The
    // downstream HKL generator branches on system name, so preserve the
    // distinction for R-centred groups.
    function deriveSystem(record) {
        const cs = record.crystal_system;
        if (cs !== 'trigonal') return cs;
        const sym = record.standard_symbol || '';
        return sym.trim().charAt(0).toUpperCase() === 'R' ? 'rhombohedral' : 'trigonal';
    }

    // 2. Symbol normalisation.
    //    The user may type "P 21 21 21", "P212121", "p 2_1 2_1 2_1",
    //    "Fd-3m", "Fd3m", "P63/mmc", "R -3 c :H", etc. Strip everything
    //    that doesn't carry meaning and uppercase the first char of each
    //    "word" for matching purposes.
    function normSymbol(s) {
        if (typeof s != 'string') return '';
        return s
            .replace(/_/g, '')             // 2_1 -> 21
            .replace(/\s+/g, '')           // drop spaces
            .replace(/:[a-zA-Z0-9]+$/, '') // drop :H / :R / :1 / :2 origin tags
            .toUpperCase();
    }

    // Rhombohedral-axes (":R") settings are not supported: the app's
    // d-spacing math and lattice UI assume hexagonal axes for R groups,
    // and the :R reflection conditions differ from :H. They are therefore
    // never indexed; resolve() maps ":R" input to the hexagonal setting
    // with a console warning.
    function isRhombAxesSetting(setting) {
        const sym = (setting && setting.symbol) ? String(setting.symbol) : '';
        const hall = (setting && setting.hall) ? String(setting.hall) : '';
        return /:R$/i.test(sym.replace(/\s+/g, '')) || hall.indexOf('*') !== -1;
    }

    // "Short" Hermann-Mauguin symbol: drop positions that are just "1".
    //   "P 1 21/c 1"   -> "P 21/c"   -> normalised "P21/C"
    //   "P 1 2 1"      -> "P 2"      -> "P2"
    //   "Fd-3m"        -> "Fd-3m"    (no spaces; returned unchanged)
    // Input may or may not have spaces; we work from whatever we've got.
    function shortSymbol(raw) {
        if (typeof raw !== 'string') return '';
        const s = raw.replace(/_/g, '').replace(/:[a-zA-Z0-9]+$/, '');
        // If the symbol has interior spaces, treat them as position delimiters
        // and drop any position that is exactly "1".
        if (/\s/.test(s.trim())) {
            const parts = s.trim().split(/\s+/);
            const kept = [parts[0]];  // centring letter
            for (let i = 1; i < parts.length; i++) {
                if (parts[i] !== '1') kept.push(parts[i]);
            }
            return kept.join('').toUpperCase();
        }
        // No spaces: nothing to shorten safely.
        return s.toUpperCase();
    }

    // The cctbx standard_symbol has embedded spaces (e.g. "P 21 21 21");
    // keep a pre-computed index for fast lookups.
    const BY_NUMBER = {};          // number -> primary setting record
    const BY_SYMBOL = new Map();   // normalised symbol -> setting record
    const BY_HALL   = new Map();   // normalised hall -> setting record
    const ALL_STANDARD_SYMBOLS = [];

    function pickPrimarySetting(sgEntry) {
        // Prefer a setting marked 'standard'. For monoclinic, none are; in
        // that case prefer the 'b' (unique axis b) setting, which is what
        // the HKL generator assumes. Otherwise fall back to the first.
        const settings = (sgEntry.settings || []).filter(s => !isRhombAxesSetting(s));
        let s = settings.find(x => x.description === 'standard');
        if (s) return s;
        // cctbx qualifiers are "b1", "-b2", "c1" and so on -- the letter is
        // the unique axis. The old test was `=== 'b'`, which never matched
        // anything, so monoclinic groups silently fell through to whichever
        // setting happened to sort first.
        s = settings.find(x => /^-?b/i.test(x.description || ''));
        if (s) return s;
        return settings[0];
    }

    function buildRecord(sgEntry, setting) {
        const laue = POINT_GROUP_TO_LAUE[sgEntry.point_group] || null;
        if (!laue) {
            // Should not happen for the 230 standard groups, but be defensive.
            console.warn('SG_ENGINE: unknown point group', sgEntry.point_group,
                         'for SG', sgEntry.number);
        }
        return {
            number: sgEntry.number,
            symbol: setting.symbol,
            standard_symbol: sgEntry.standard_symbol,
            setting_description: setting.description,
            hall: setting.hall,
            system: deriveSystem(sgEntry),
            laue_class: laue,
            point_group: sgEntry.point_group,
            centering: (setting.symbol || '').charAt(0).toUpperCase(),
            centrosymmetric: !!sgEntry.centrosymmetric,
            reflection_conditions: setting.reflection_conditions || {}
        };
    }

    (function buildIndices() {
        Object.keys(DB).forEach(key => {
            const sgEntry = DB[key];
            const primary = pickPrimarySetting(sgEntry);
            if (!primary) return;
            const primaryRec = buildRecord(sgEntry, primary);
            BY_NUMBER[sgEntry.number] = primaryRec;
            ALL_STANDARD_SYMBOLS.push(sgEntry.standard_symbol);

            // Index every setting of every group by its symbol, Hall name,
            // AND the short-form symbol ("P 1 21/c 1" -> "P21/C").
            (sgEntry.settings || []).forEach(setting => {
                if (isRhombAxesSetting(setting)) return;  // never index :R settings
                const rec = buildRecord(sgEntry, setting);
                const sKey = normSymbol(setting.symbol);
                if (sKey && !BY_SYMBOL.has(sKey)) BY_SYMBOL.set(sKey, rec);
                const shortKey = normSymbol(shortSymbol(setting.symbol));
                if (shortKey && !BY_SYMBOL.has(shortKey)) BY_SYMBOL.set(shortKey, rec);
                const hKey = normSymbol(setting.hall);
                if (hKey && !BY_HALL.has(hKey)) BY_HALL.set(hKey, rec);
            });
            // Also map the canonical "standard_symbol" (and its short form)
            // to the primary record. Short form works because the
            // standard_symbol is stored with spaces, e.g. "P 1 21/c 1".
            const stdKey = normSymbol(sgEntry.standard_symbol);
            if (stdKey && !BY_SYMBOL.has(stdKey)) BY_SYMBOL.set(stdKey, primaryRec);
            const stdShort = normSymbol(shortSymbol(sgEntry.standard_symbol));
            if (stdShort && !BY_SYMBOL.has(stdShort)) BY_SYMBOL.set(stdShort, primaryRec);
        });
    })();

    function resolve(input) {
        if (input == null) return null;
        // Number, or numeric string
        if (typeof input === 'number' && Number.isFinite(input)) {
            return BY_NUMBER[input] || null;
        }
        if (typeof input === 'string') {
            const trimmed = input.trim();
            if (/^\d+$/.test(trimmed)) {
                const n = parseInt(trimmed, 10);
                return BY_NUMBER[n] || null;
            }
            // Rhombohedral-axes input is not supported; fall through to the
            // hexagonal setting (normSymbol strips the :R tag) but say so.
            if (/:R$/i.test(trimmed.replace(/\s+/g, ''))) {
                console.warn('SG_ENGINE: rhombohedral-axes setting ":R" is not supported; resolving to the hexagonal-axes setting.');
            }
            // First try the user's input verbatim (normalised).
            const key = normSymbol(trimmed);
            const direct = BY_SYMBOL.get(key) || BY_HALL.get(key);
            if (direct) return direct;
            // Then try the short form (drops stand-alone "1" positions).
            const shortKey = normSymbol(shortSymbol(trimmed));
            return BY_SYMBOL.get(shortKey) || null;
        }
        return null;
    }

    // 3. Reflection-condition evaluator.
    function inFamily(h, k, l, family) {
        switch (family) {
            case 'hkl': return true;
            case 'h00': return k === 0 && l === 0;
            case '0k0': return h === 0 && l === 0;
            case '00l': return h === 0 && k === 0;
            case 'h0l': return k === 0;
            case '0kl': return h === 0;
            case 'hk0': return l === 0;
            case 'hhl': return Math.abs(h) === Math.abs(k);
            default:    return false;
        }
    }

    // Parses once, returns a closure over (h,k,l). Cached per SG record.
    const CONDITION_RE = /^\s*(-?\d*[a-z](?:\s*[+\-]\s*\d*[a-z])*)\s*=\s*(\d+)\s*n\s*$/i;

    function parseExpression(expr) {
        // Return a function (h,k,l) -> integer
        // Accept tokens like: h, k, l, 2h, -h, +h, 2l
        const clean = expr.replace(/\s+/g, '');
        // Split into signed terms: split on '+' or '-' keeping the sign
        const tokens = clean.match(/[+\-]?\d*[hkl]/g);
        if (!tokens) return null;
        const parts = tokens.map(tok => {
            const m = tok.match(/^([+\-]?)(\d*)([hkl])$/);
            if (!m) return null;
            const sign = m[1] === '-' ? -1 : 1;
            const coeff = m[2] === '' ? 1 : parseInt(m[2], 10);
            return { sign, coeff, var: m[3] };
        });
        if (parts.some(p => p === null)) return null;
        return function (h, k, l) {
            let total = 0;
            for (const p of parts) {
                const v = p.var === 'h' ? h : p.var === 'k' ? k : l;
                total += p.sign * p.coeff * v;
            }
            return total;
        };
    }

    function parseCondition(str) {
        // The cctbx export writes some coefficients with an explicit product,
        // e.g. the d-glide condition "2*h+l=4n" on hhl. CONDITION_RE has no
        // room for the '*', so those rules used to fail to parse and were
        // silently DROPPED -- letting through reflections that are in fact
        // systematically absent in I41md, I41cd, I-42d, I41/amd, I41/acd,
        // I-43d and Ia-3d. parseExpression already handles "2h", so the '*'
        // just needs removing first.
        const m = str.replace(/\*/g, '').match(CONDITION_RE);
        if (!m) {
            console.warn('SG_ENGINE: unparsable reflection condition:', str);
            return null;
        }
        const fn = parseExpression(m[1]);
        const divisor = parseInt(m[2], 10);
        if (!fn || !divisor) return null;
        return { fn, divisor, raw: str };
    }

    // Compile reflection_conditions into parsed form on first use; cache
    // on the record itself to avoid reparsing on every HKL.
    function compileConditions(sg) {
        if (sg.__compiled_conditions) return sg.__compiled_conditions;
        const out = {};
        const src = sg.reflection_conditions || {};
        for (const fam in src) {
            const parsed = [];
            for (const rule of src[fam]) {
                const p = parseCondition(rule);
                if (p) parsed.push(p);
            }
            out[fam] = parsed;
        }
        Object.defineProperty(sg, '__compiled_conditions', {
            value: out, enumerable: false, writable: false
        });
        return out;
    }

    function isReflectionAllowed(h, k, l, sg) {
        if (!sg) return true;
        const conds = compileConditions(sg);
        // For every family the hkl is a member of, ALL conditions must hold.
        for (const fam in conds) {
            if (!inFamily(h, k, l, fam)) continue;
            const rules = conds[fam];
            for (const r of rules) {
                const val = r.fn(h, k, l);
                if (val % r.divisor !== 0) return false;
            }
        }
        return true;
    }

    //   Multiplicities from Laue orbits (replaces the hand-written table).
    //
    //   The old switch was a lookup table keyed on |h|, |k|, |l|. Three of
    //   its branches were wrong, and each error was masked by the HKL
    //   generator, which only ever fed it indices from an h>=k>=l>=0 or
    //   h>=k>=0 wedge:
    //
    //     6/mmm  {h,-2h,l} (e.g. (1,-2,l), (2,-4,l)) came out 24; it is 12.
    //            |h| and |k| are not the hexagonal invariants -- the third
    //            Bravais-Miller index i = -(h+k) is on a mirror whenever
    //            h = i or k = i, which |h| vs |k| cannot see.
    //     m-3    {hhl} and {hkk} came out 12; they are 24. m-3 has no mirror
    //            swapping two axes, so those forms are general, not special.
    //     -3m    h0l / hhl were given the same value. They differ, and which
    //            one is special depends on the setting.
    //
    //   Computing the orbit under the actual Laue operators removes the whole
    //   class of error. Operators are applied as (h,k,l) -> (h,k,l)*R, the
    //   same convention used elsewhere in this project.
    //
    //   `sg` is optional and only consulted for -3m: pass the space-group
    //   number (or a record with .number) to pick the right setting. Without
    //   it, -3m1 is assumed, which covers 12 of the 19 groups involved.

    const LAUE_GENERATORS = {
        '-1':    [],
        '2/m':   [[-1, 0, 0, 0, 1, 0, 0, 0, -1]],                                   // 2 || b (default)
        '2/m_a': [[1, 0, 0, 0, -1, 0, 0, 0, -1]],                                   // 2 || a
        '2/m_b': [[-1, 0, 0, 0, 1, 0, 0, 0, -1]],                                   // 2 || b
        '2/m_c': [[-1, 0, 0, 0, -1, 0, 0, 0, 1]],                                   // 2 || c
        'mmm':   [[1, 0, 0, 0, -1, 0, 0, 0, -1], [-1, 0, 0, 0, 1, 0, 0, 0, -1]],
        '4/m':   [[0, -1, 0, 1, 0, 0, 0, 0, 1]],
        '4/mmm': [[0, -1, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, -1, 0, 0, 0, -1]],
        '-3':    [[0, -1, 0, 1, -1, 0, 0, 0, 1]],
        '-3m1':  [[0, -1, 0, 1, -1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, -1]],     // 2 || <100>
        '-31m':  [[0, -1, 0, 1, -1, 0, 0, 0, 1], [0, -1, 0, -1, 0, 0, 0, 0, -1]],   // 2 || <1-10>
        '6/m':   [[1, -1, 0, 1, 0, 0, 0, 0, 1]],
        '6/mmm': [[1, -1, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, -1]],
        'm-3':   [[-1, 0, 0, 0, -1, 0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 1, 1, 0, 0, 0, 1, 0]],
        'm-3m':  [[0, -1, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 0]]
    };

    // P312, P3112, P3212, P31m, P31c, P-31m, P-31c -- the trigonal groups
    // whose symmetry sits on the tertiary directions. All other groups in
    // 149-167 are -3m1.
    const LAUE_31M_NUMBERS = { 149: 1, 151: 1, 153: 1, 157: 1, 159: 1, 162: 1, 163: 1 };

    function matMul3(A, B) {
        const C = new Array(9);
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
            }
        }
        return C;
    }

    const LAUE_GROUP_CACHE = {};

    function laueGroup(key) {
        if (LAUE_GROUP_CACHE[key]) return LAUE_GROUP_CACHE[key];
        const gens = LAUE_GENERATORS[key];
        if (!gens) return null;
        const map = new Map();
        const push = (m) => {
            const k = m.join(',');
            if (map.has(k)) return false;
            map.set(k, m);
            return true;
        };
        push([1, 0, 0, 0, 1, 0, 0, 0, 1]);
        push([-1, 0, 0, 0, -1, 0, 0, 0, -1]);      // Laue class = point group + -1
        gens.forEach(push);
        for (let pass = 0; pass < 8; pass++) {
            const cur = Array.from(map.values());
            let grew = false;
            for (let i = 0; i < cur.length; i++) {
                for (let j = 0; j < cur.length; j++) {
                    if (push(matMul3(cur[i], cur[j]))) grew = true;
                    if (map.size > 48) return null;
                }
            }
            if (!grew) break;
        }
        LAUE_GROUP_CACHE[key] = Array.from(map.values());
        return LAUE_GROUP_CACHE[key];
    }

    function laueKey(laue_class, sg) {
        if (laue_class === '-3m') {
            const n = (sg && typeof sg === 'object') ? sg.number : sg;
            return LAUE_31M_NUMBERS[n] ? '-31m' : '-3m1';
        }
        if (laue_class === '2/m') {
            // Monoclinic multiplicities depend on which axis is unique:
            // (h,k,0) has m = 2 on unique axis b but m = 4 on unique axis c.
            // The setting qualifier ("b1", "-b2", "c1", ...) carries it.
            const d = (sg && typeof sg === 'object') ? sg.setting_description : null;
            const m = (typeof d === 'string') ? d.match(/^-?([abc])/i) : null;
            return m ? '2/m_' + m[1].toLowerCase() : '2/m';
        }
        return laue_class;
    }

    function getMultiplicity(h, k, l, laue_class, sg) {
        if (h === 0 && k === 0 && l === 0) {
            return { multiplicity: 1, canonical_hkl_obj: [0, 0, 0] };
        }
        const ops = laueGroup(laueKey(laue_class, sg));
        if (!ops) {
            console.warn('SG_ENGINE: unknown Laue class', laue_class, '-> m=1');
            return { multiplicity: 1, canonical_hkl_obj: [h, k, l] };
        }
        const seen = new Set();
        let ch = -Infinity, ck = 0, cl = 0;
        for (let i = 0; i < ops.length; i++) {
            const r = ops[i];
            const H = h * r[0] + k * r[3] + l * r[6];
            const K = h * r[1] + k * r[4] + l * r[7];
            const L = h * r[2] + k * r[5] + l * r[8];
            seen.add(H + ',' + K + ',' + L);
            if (H > ch || (H === ch && (K > ck || (K === ck && L > cl)))) { ch = H; ck = K; cl = L; }
        }
        return { multiplicity: seen.size, canonical_hkl_obj: [ch, ck, cl] };
    }

    // 5. Helpers exposed for the UI: list of all 230 standard symbols
    //    for <datalist> autocomplete, and an "is supported" probe.
    function list() {
        return ALL_STANDARD_SYMBOLS.slice();
    }


    function listSettings(n) {
    const db = globalThis.SG_DATABASE;
    if (!db || !db.space_groups) return [];
    const sg = db.space_groups[n.toString()];
    if (!sg) return [];

    return sg.settings.filter(s => !isRhombAxesSetting(s)).map(setting => ({
        number: sg.number,
        standard_symbol: sg.standard_symbol,
        system: deriveSystem(sg),   // FIX: was raw crystal_system; R groups need 'rhombohedral'

        point_group: sg.point_group,
        laue_class: POINT_GROUP_TO_LAUE[sg.point_group] || sg.point_group,
        centrosymmetric: sg.centrosymmetric,
        symbol: setting.symbol,
        setting_description: setting.description,
        hall: setting.hall,
        centering: setting.symbol.charAt(0),
        reflection_conditions: setting.reflection_conditions
    }));
}

    root.SG_ENGINE = {
        resolve:              resolve,
        isReflectionAllowed:  isReflectionAllowed,
        getMultiplicity:      getMultiplicity,
        list:                 list,
        listSettings:         listSettings,
        // Exposed for tests only.
        _parseCondition:      parseCondition,
        _inFamily:            inFamily,
        _POINT_GROUP_TO_LAUE: POINT_GROUP_TO_LAUE
    };
})(typeof self !== 'undefined' ? self : globalThis);
