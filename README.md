# comp-Harker

**Composition-driven Harker/Patterson structure solution**

comp-Harker is a browser-based crystallographic structure-solution tool. It combines:

- 3D FFT Patterson-map synthesis
- Harker-section analysis
- Symmetry Minimum Function (SMF) and Buerger superposition
- WebGPU-accelerated replica-exchange Monte Carlo
- Composition-driven Wyckoff-position searches
- Patterson map correlation and crystallographic/chemical candidate checks

Earlier versions were called **Harko** and **sHarko**. The composition-driven route was added in August 2026, when the project was renamed **comp-Harker**.

> This README is a simplified version of the full technical help file. It focuses on what users need to understand, run, inspect, and troubleshoot the program.

## What comp-Harker does

The program starts from diffraction intensities and constructs a **Patterson map**. Patterson peaks represent interatomic displacement vectors rather than direct atomic positions.

The basic relationship is:

$$
P(\mathbf{u}) =
\frac{1}{V}\sum_{\mathbf{h}} |F_{\mathbf{h}}|^2
\cos(2\pi\mathbf{h}\cdot\mathbf{u})
$$

Because Patterson maps are centrosymmetric, peaks at `u` and `-u` are equivalent.

For structures with symmetry, **Harker sections** constrain certain interatomic vectors to planes or lines. These constraints can be used to obtain candidate atomic positions.

The current structure-solution route is **composition-driven**: provide a chemical formula and the program searches combinations of Wyckoff positions whose multiplicities reproduce that composition exactly.

## Main workflow

```text
Diffraction data
      |
      v
Intensity correction
      |
      v
Patterson FFT
      |
      +--> Patterson peaks
      |
      +--> Harker sections
      |
      +--> SMF / Buerger superposition
      |
      v
Candidate atomic sites
      |
      v
Formula + Z + constraints
      |
      v
Wyckoff assignment search
      |
      v
WebGPU replica-exchange Monte Carlo
      |
      v
Candidate structures
      |
      +--> R factor
      +--> Patterson correlation
      +--> Contact checks
      +--> Coordination
      +--> Bond-length spread
      +--> Bond-valence
      +--> Hamilton significance
      |
      v
CIF / PDF / GRD / CSV
```

---

## 1. Input data

comp-Harker reads ASCII diffraction exports such as:

- Powder diffraction refinement exports (for example, Powder 5 output)
- Generic peak lists
- SHELX HKLF 4-style data (`h, k, l, intensity, sigma`)

HKLF 4 support exists but was not fully tested in the original documentation.

If the input file contains metadata such as the space group or unit-cell parameters, comp-Harker attempts to populate those fields automatically.

### Intensity correction

The Patterson calculation ideally uses `|F|²`, not raw powder peak areas. Raw intensities may contain:

- Lorentz-polarization effects (`Lp`)
- Powder multiplicity (`m`)

The program chooses the best available input route:

| Priority | Available data | Calculation |
|---|---|---|
| 1 | `|Fo|` | `|F|² = |Fo|²` |
| 2 | `I_hkl`, `m`, `Lp` | `|F|² = I_hkl / (m · Lp)` |
| 3 | `I_hkl`, `2θ` | `|F|² = I_hkl / Lp(2θ)` |
| 4 | `I_hkl` only | `|F|² = I_hkl` |

The last case is a fallback. Without corrections, low-angle reflections can dominate the Patterson map.

If the data contain `d` and intensity but no wavelength, the program refuses the conversion because it cannot reliably determine `2θ`.

---

## 2. Patterson map calculation

The map is calculated with a 3D Radix-2 FFT:

```text
Expand reflections using symmetry
        ↓
Apply optional Lorch filtering
        ↓
Choose power-of-two FFT grid
        ↓
Populate reciprocal-space grid
        ↓
3D inverse FFT
        ↓
Normalize by cell volume
        ↓
Find peaks / calculate SMF / Buerger map
```

The FFT scales as approximately `O(N³ log N)`.

### Lorch filter

The Lorch strength `s` ranges from `0` to `1`.

- `0`: no filtering
- intermediate values: blend raw data with the Lorch envelope
- `1`: full Lorch filtering

Increasing the filter suppresses termination ripples but can broaden peaks.

### Grid resolution

The FFT grid is automatically increased to a suitable power of two. It must satisfy:

```text
N >= 2 * h_max + 1
```

This prevents high-index reflections from aliasing onto lower-index data.

The UI exposes 64³ and 128³ as the normal choices, but the program can increase the grid automatically if required by the data.

### Calculated maps

For model structures, calculated Patterson maps are broadened to match the experimental resolution. Peak width and the overall temperature factor `B` affect the calculated and difference maps, not the observed map or search fitness.

---

## 3. Peak and Harker analysis

### Patterson peaks

The program finds local maxima using a 26-neighbour 3D search with periodic boundary wrapping.

The origin is masked in the raw Patterson map because the origin contains the strong self-vector contribution.

### Harker sections

Space-group symmetry operations constrain some vectors to special planes or lines.

For an operation

```text
Sx = Rx + t
```

the corresponding Harker vector is

```text
uH = x - (Rx + t)
```

or

```text
uH = (I - R)x - t
```

Peaks near these sections can therefore be converted into candidate fractional coordinates.

### Consolidated sites

The program currently has two ways to obtain the consolidated candidate-site list:

1. **SMF / Buerger superposition**
2. **Harker-section combination**, when enabled and enough section solutions exist

The raw Harker solutions are retained as useful diagnostic information. The current SMF/superposition route does not simply average all Harker solutions together.

### SMF

The Symmetry Minimum Function evaluates the Patterson intensity predicted by all non-identity symmetry operations:

$$
SMF(\mathbf{x}) =
\min_i P\left(
\mathbf{x}-(R_i\mathbf{x}+t_i)
\right)
$$

This acts like a logical **AND** filter. Genuine atomic sites can survive while unrelated Patterson-vector overlaps are suppressed.

SMF positions are expressed in the absolute crystallographic frame.

### Buerger fallback

For `P1`, or when symmetry operators are unavailable, the program uses a Buerger minimum superposition:

$$
M(\mathbf{u}) =
\min\left(P(\mathbf{u}),P(\mathbf{u}-\mathbf{u}_{top})\right)
$$

Unlike SMF, this fixes the origin relative to the selected strong vector.

---

## 4. Composition-driven structure search

This is the current main solution method.

Enter:

- chemical formula
- space group
- unit-cell parameters
- optionally `Z`
- any justified distance constraints

For example:

```text
Formula: PbSO4
```

The search determines:

- how many atoms are required
- which Wyckoff positions can contain them
- the free coordinates of those positions
- the coordinates that best fit the diffraction data

### Why Wyckoff positions?

A general-position search gives every independent atom three free coordinates.

A Wyckoff-position search instead uses the symmetry constraints built into the position.

For example, a site may have:

```text
(x, 1/4, z)
```

instead of three independent coordinates, or:

```text
(0, 0, 0)
```

with no free coordinates.

This reduces the search space while enforcing the requested composition.

### Exact composition

Every enumerated structure must satisfy the requested stoichiometry by construction.

The database contains Wyckoff projectors and coset operators generated from cctbx data. The documentation reports:

- 230 space groups
- 527 settings
- exact Wyckoff multiplicity handling

### `Z`

`Z` may be supplied explicitly or left blank.

When omitted, comp-Harker estimates a plausible range from cell volume and flags values related to the space-group order. This is only a suggestion, not a determination.

A wrong `Z` can make it impossible to construct the requested composition from available Wyckoff multiplicities.

---

## 5. Search algorithm

The structure search uses **replica-exchange Monte Carlo** on the GPU.

Each chain contains one trial structure.

At each step:

1. Propose a Gaussian move in the free coordinates.
2. Project the coordinates back onto the selected Wyckoff subspace.
3. Evaluate the structure.
4. Accept an improvement automatically.
5. Sometimes accept a worse structure using the Metropolis rule.

$$
P(\mathrm{accept}) =
\min\left[
1,\exp\left(\frac{f_{new}-f_{cur}}{T}\right)
\right]
$$

The temperature is fixed for each ladder rung rather than cooled continuously.

There are eight temperature rungs, geometrically spanning approximately:

```text
5 × 10⁻⁴  →  0.05
```

Adjacent rungs attempt exchanges every 10 steps.

The purpose of the hot chains is to explore broadly while the cold chains refine promising structures.

### Final quench

After the stochastic search, every assignment is refined using a greedy `T = 0` descent:

- only improvements are accepted
- the step size decreases geometrically
- the calculation uses full resolution and penalty weight

This means the final candidate table is based on structures that have received a final local refinement.

---

## 6. Search controls

### Chains

Number of independent Markov chains.

More chains explore more starting regions, but they do **not** replace sufficiently long runs.

The maximum is determined by the connected GPU's WebGPU limits.

### Iterations

Number of steps per chain.

This is usually the most important search-depth control. High-dimensional Wyckoff assignments need enough iterations to explore their free parameters.

Default documented value:

```text
4000
```

Maximum documented value:

```text
10000
```

### Restarts

Fresh random starts while retaining the best structure already found.

Restarts improve independence between attempts; iterations allow each attempt to explore more deeply.

### Practical rule

First increase **Chains** enough to avoid excessive waves. Then spend additional budget on **Iterations**.

Too few chains can force the assignment set into multiple waves, substantially increasing wall time because each wave repeats the search budget.

---

## 7. GPU limits

Two separate hardware limits matter.

### Number of chains

The program examines WebGPU storage-buffer and workgroup limits and uses them to determine how many particles can run simultaneously.

### Atoms × symmetry operators

A single particle must hold all symmetry-generated atomic positions required for its fitness calculation.

Therefore:

```text
asymmetric-unit atoms × space-group operators
```

must fit in the GPU's workgroup storage.

If the structure is too large for the device, comp-Harker refuses to start the run rather than silently dropping atoms.

The error reports the requested and available limits so the cause is explicit.

---

## 8. Contact and distance constraints

### Minimum contact distance

A global hard floor applies to every interatomic contact, including symmetry mates and cell-boundary neighbours.

Range:

```text
0–3 Å
```

Default:

```text
1.00 Å
```

`0` disables the floor.

Raise it carefully: setting the value too high can exclude the correct structure.

### Distance windows

Constraints can be entered per line.

Example:

```text
S O 1.35 1.65
```

This means:

- S–O must not be shorter than 1.35 Å
- each S should have an O within 1.65 Å

For a coordination-style constraint, the UI also supports the form:

```text
S O 4 1.4/1.9
```

The lower bound is a hard constraint. The upper bound is interpreted as a nearest-neighbour requirement rather than requiring every S–O pair in the cell to be inside the window.

Constraints are enforced, not merely added as ranking penalties.

---

## 9. Search fitness and candidate ranking

A key distinction:

> **The search maximises Patterson map correlation, but the final candidate table ranks structures using the R factor.**

### Patterson correlation

The current search fitness is the correlation between observed and calculated Patterson information.

This is useful for finding structures because it is sensitive to the interatomic-vector pattern.

The search therefore uses correlation as its optimisation target.

### R factor

The final candidates are ranked using `R` against `|F|`.

This gives weak reflections and light-atom information more influence than the Patterson correlation alone.

This distinction matters because heavy-atom vectors can make several chemically different structures have very similar Patterson correlations.

### Hamilton's test

Models with more free parameters can fit data better simply because they are more flexible.

Hamilton's R-ratio test is therefore used to ask whether the improvement in fit is statistically meaningful.

The UI reports:

- `p < 0.05`: top model is genuinely better
- `n.s.`: models are not distinguishable by this data
- `equal dim`: same number of parameters; test does not apply
- `dominated`: a more complex model fits worse

The documentation notes that the reported `p` values are indicative because the implementation uses unit weights when observation uncertainties are unavailable.

---

## 10. How to judge candidates

Do not choose a structure from correlation alone.

The **Structure quality** panel provides:

### Coordination number

Counts atoms in the first coordination shell.

### Bond-length spread

Measures how uniform the first-shell distances are.

A small spread can indicate a regular coordination environment, although it should not be used alone.

### Bond-valence sum

Uses the bond-valence relationship

$$
\sum_j \exp\left(\frac{R_0-d_j}{0.37}\right)
$$

and compares the result with plausible oxidation states.

Unlisted element pairs are not guessed.

### R factor

Provides an independent diffraction-fit measure.

### Best practice

Consider all of these together:

```text
Patterson correlation
        +
R factor
        +
Hamilton significance
        +
bond lengths
        +
coordination
        +
bond valence
        +
distance constraints
```

A model with excellent correlation can still be chemically implausible.

---

## 11. UI overview

The interface has two main areas.

### Left panel

Five collapsible sections:

1. **Data Input**
2. **Structure**
3. **Patterson Map Synthesis**
4. **Structure Search**
5. **Log**

Controls provide `?` help boxes with explanations and current state information.

### Right panel

Three tabs:

#### Maps

Shows three square panes:

- **Observed**
- **Calculated**
- **Difference**

The SMF map is still calculated but no longer has its own pane.

Controls include:

- Auto range
- Axis
- Section
- Density export

#### Peaks

Shows:

1. Patterson peaks
2. Harker solutions
3. Consolidated sites

Export with **Peaks (.csv)**.

#### Search Results

Contains:

- fitness evolution
- 3D structure viewer
- atom count and shortest contact
- optional `B` fitting
- candidate structures
- asymmetric-unit coordinates
- structure-quality information

Clicking a candidate loads it into the associated views.

---

## 12. Recommended workflows

### A. Deterministic Harker / SMF analysis

Use this when you first want to inspect the Patterson information without running the GPU search.

1. Load the diffraction file.
2. Select and verify the space group.
3. Verify the unit cell.
4. Set Lorch/grid/peak options.
5. Click **Calculate Map**.
6. Inspect the observed Patterson map.
7. Review Patterson peaks and Harker solutions.
8. Compare the two consolidated-site routes.

The map calculation and peak analysis are deterministic.

### B. Composition-driven global search

1. Load and calculate the Patterson map.
2. Enter the chemical formula.
3. Set `Z` if known.
4. Add justified distance constraints.
5. Set the global minimum contact distance.
6. Increase Chains until the search avoids unnecessary waves.
7. Use Iterations for search depth.
8. Add Restarts when additional independent searches are useful.
9. Run the search.
10. Inspect candidate structures using chemistry and `R`, not correlation alone.
11. Export the selected structure.

Heavy-atom seeding from consolidated Harker/SMF sites is automatic.

### C. Hybrid workflow

For a normal structure solution:

```text
Load data
  ↓
Calculate Patterson map
  ↓
Inspect peaks / Harker / SMF
  ↓
Enter formula and constraints
  ↓
Run composition-driven search
  ↓
Compare candidates
  ↓
Validate chemistry and R
  ↓
Export CIF / report
```

---

## 13. Exports

### CIF

**Save CIF** writes the selected candidate with:

- unit cell
- space group
- solved coordinates
- occupancy `1.0`
- Wyckoff multiplicity

### GRD

**Density (.grd)** writes VESTA-readable maps, including:

- observed
- calculated
- difference
- SMF

### CSV

**Peaks (.csv)** contains:

- Patterson peaks
- Harker solutions
- consolidated sites

### PDF

**PDF Report** includes the main structural parameters, maps, fitness plot, peak/site tables, and structure information.

---

## 14. Troubleshooting

### WebGPU is not supported

Use a Chromium-based browser with WebGPU support and up-to-date graphics drivers. The original documentation specifies Chromium 113+.

### The search refuses to start because of a GPU limit

The asymmetric-unit atom count multiplied by the number of symmetry operators exceeds the GPU's workgroup-storage limit.

Reduce the structure size if possible or use a GPU with more workgroup storage.

### Every particle has the same score

The compute kernel may have dispatched but failed to write its results.

Check the browser console for WGSL validation errors and perform a hard reload.

### Candidates are all within ~0.01 correlation

This can be normal. Patterson correlation may not distinguish light-atom arrangements well.

Use:

- structure quality
- bond lengths
- bond valence
- `R`
- Hamilton's test

Also consider increasing Iterations and Restarts.

### Every candidate has an unbonded cation

The search may have no chemical information telling it to form the expected polyatomic group.

Add an appropriate distance window, for example:

```text
S O 1.35 1.65
```

### No Wyckoff assignment matches the requested composition

Check `Z`.

The composition may require a different `Z`, or may require partial occupancy, which this search does not model.

### Wilson B is reported as unphysical

This usually indicates that intensity correction or sharpening has been applied twice.

Check the intensity route reported in the Log.

### Old or missing Wyckoff data

Regenerate the space-group database with:

```text
cctbx_Harko_v1.py
```

Check which database file the loader selected and whether the settings contain the expected projection fields.

### Stale JavaScript modules

Keep the modules and `Harko.html` together and reload with:

```text
Ctrl+Shift+R
```

The application checks module version strings and reports mismatches.

---

## 15. Developer architecture

The main computational layers are:

| Layer | Role |
|---|---|
| Main UI thread | File parsing, UI state, Three.js rendering, map display, PDF reports |
| Web Worker | FFT, Lorch filtering, peak finding, SMF/Buerger maps, Harker analysis |
| WebGPU | Structure fitness, symmetry expansion, contact evaluation |

Important modules documented by the project include:

```text
Harko.html
patterson3d.js
style.css
sg_engine.js
sharko_worker.js
symmetry_utils.js
wyckoff_assign.js
observations.js
scatterers.js
contacts.js
swarm_wyckoff.js
swarm_compute.wgsl
swarm_cc.wgsl
```

Data directories include:

```text
sg/
scatters/
```

The space-group database is generated by:

```text
cctbx_Harko_v1.py
```

Scattering-factor data are generated by:

```text
cctbx_scatterers_v1.py
```

Modules carry version strings that are checked by the main HTML application. Keep matching module versions together.

---

## 16. Important implementation notes

- The Patterson origin is masked during raw-map peak analysis.
- SMF/superposition peak detection is unmasked so valid sites on special positions are not removed.
- The SMF map is still calculated even though its dedicated UI pane was removed.
- Harker combination is only meaningful when at least two sections provide partial sites.
- Changing most map controls marks the map as out of date; **Calculate Map** must be pressed explicitly.
- Harker tolerances operate on existing peaks and can update site lists immediately.
- Search, map calculation, file loading, and exports are locked while a calculation is running, except for **Stop**.
- Candidate coordinates are read-only because they are search results; dependent displays are derived from them.
- The current sampler is replica-exchange Monte Carlo, not the older particle-swarm implementation.
- The older vector/sigma fitness is retained in the technical documentation for historical context; the current search uses map correlation.

---

## Version

The original Harko version dates to **12 October 2025**.

The composition-driven solution was added in **August 2026**, when the application became **comp-Harker**.

Documentation source: `comp_harko_help.html`, dated 7 August 2026 in the source file.

This README file was generated by an AI on August 12, 2026.
