# Graphical modeling design notes (expanded)

## Overview
This document captures:
1) the current model definition UI and input-mode code paths,
2) the target UX for a click-first Graphical modeling method,
3) an authoritative mapping from the 2D grid editor to existing JointConfiguration primitives,
4) migration/backward-compatibility and a realistic test plan (Streamlit-friendly).

Primary requirement: users must be able to define a model primarily by mouse clicks on the grid (nodes/edges), with parameter editing via contextual popovers and/or the Model Parameters tab (tables are for review/edit, not mandatory for creation).

---

## Goals / Non-goals

### Goals
- Keep ONLY Input Mode = Standard. Remove Refined Row and Node-based from UI and code paths.
- Add Method = Graphical (under Standard input mode).
- Add a new tab "Modeling" next to "Scheme" in Model Definition view.
- Modeling tab: interactive grid editor with click-first creation/deletion of:
  - node
  - plate segments
  - fastener segments / fastener rows
  - supports
  - loads
- Bottom area becomes two tabs:
  - Model Parameters (minimalistic engineering UI, bulk edits)
  - Results (unchanged behavior)
- Sidebar cleanup:
  - remove geometry definition controls from sidebar
  - keep: input mode/method, dimension system, config loading, available cases + actions
  - move navigation switches (Model Definition / Model Comparison / Export/Reports) into sidebar
- Preserve solver/export/case management/comparison/reporting functionality.

### Non-goals (for the first implementation)
- Introducing new solver entities or changing solver contracts.
- Complex drag-and-drop or freehand drawing. (Click-to-toggle is sufficient.)
- Full CAD-like snapping, arbitrary geometry, or non-grid-based modeling.

---

## Current UI state (as observed)

### Input modes: where they live and how they behave today
- Definitions: input mode is selected in `jolt/ui/components.render_sidebar` via a selectbox with `Standard`, `Refined Row`, and `Node-based`.
- State: stored in `st.session_state.input_mode` (widget key `input_mode_selector`).
- Rendering paths:
  - Standard / Refined Row: layered geometry form:
    `_render_geometry_section` (rows & pitches),
    `_render_plates_section`,
    `_render_fasteners_section`,
    `_render_supports_section`.
  - Refined Row adds an "Extra Nodes" expander and calls `jolt.inputs.process_refined_rows`.
  - Node-based renders editable tables via `_render_node_based_inputs`, converted through `jolt.inputs.process_node_based`.
- Serialization: `JointConfiguration` (in `jolt/config.py`) stores primitives (pitches, plates, fasteners, supports, point forces, units) but not the UI input mode. Mode is purely UI state, which makes removing modes safer.

---

## Target UX: Standard input mode + methods (Standard vs Graphical)

### Method selection
- Input Mode selector remains but contains only "Standard".
- Add Method selector with:
  - Standard (existing form-based editor)
  - Graphical (new grid editor)
- Switching methods must not lose data:
  - Graphical method edits the same JointConfiguration primitives.
  - If additional UI-only metadata is used (recommended), it must be stored as optional metadata and ignored by solver.

### Model Definition view
- Tabs: `Scheme` (existing) and `Modeling` (new).
- `Scheme`: no changes.
- `Modeling`: click-first grid editor (described below).

### Bottom area
- Tabs:
  - `Model Parameters` (new structured editor, bulk edits)
  - `Results` (existing result tables; unchanged behavior)

---

## Click-first grid editor: interaction model

### Core principle
Users create/delete the model by clicking with the mouse:
- click intersections for nodes,
- click horizontal edges for plate segments,
- click vertical edges for fastener segments,
- click nodes for supports/loads.
No table editing is required to build the topology.

### Toolbar
- Interaction mode toggle:
  - Create
  - Delete
- Active tool selector:
  1) Node
  2) Plate
  3) Fastener
  4) Support
  5) Load
- Optional: Select/Edit tool (nice-to-have; can be implicit via clicking existing entities).

### Click targets
- Node tool: click grid intersection (plate_idx, row_idx)
- Plate tool: click horizontal edge between two adjacent intersections on the same plate line:
  - edge: (plate_idx, row_idx) <-> (plate_idx, row_idx+1)
- Fastener tool: click vertical edge between two adjacent intersections on the same station/column:
  - edge: (plate_idx, row_idx) <-> (plate_idx+1, row_idx)
- Support tool: click an existing node (intersection) to add/remove support
- Load tool: click an existing node (intersection) to add/remove load

### Validations (must be strict and safe)
- Plate edge creation requires nodes at both endpoints (no auto-create nodes unless explicitly decided later).
- Fastener edge creation requires nodes at both endpoints.
- Support/load requires the node to exist.

### Feedback
- Hover highlight of the entity under cursor (node/edge).
- In Create mode: show a “ghost” preview if valid; show error tooltip if invalid.
- In Delete mode: highlight removable entities; show warning if deletion cascades.

### Deletion cascade rules
Deleting a node must remove dependent entities safely:
- remove supports/loads attached to that node,
- remove plate edges incident to that node,
- remove fastener edges incident to that node,
- update/trim corresponding Plate/FastenerRow representations.
If the node has dependencies, show a small confirm popover (“This will also remove X plate segments, Y fastener segments, Z loads/supports”).

---

## Grid coordinate system (authoritative)

We model a “joint grid” as:
- Vertical axis: plate lines / elements (index `p`, 0..P-1)
- Horizontal axis: stations along the joint (fastener row positions) (index `r`, 0..R-1)

Definitions:
- Node: N[p,r] exists at intersection of plate line p and station r.
- Plate segment: H[p,r] exists between N[p,r] and N[p,r+1] (horizontal adjacency).
- Fastener segment: V[p,r] exists between N[p,r] and N[p+1,r] (vertical adjacency).
- Pitch: pitches[r] is the spacing between station r and r+1.
  Therefore len(pitches) = R-1.

Important: This grid is topological. Physical dimensions are determined by pitches + units.

---

## Mapping grid state to existing JointConfiguration primitives

### Pitches
- `JointConfiguration.pitches` := pitches array from the grid header inputs.

### Plates (elements)
Interpretation aligned with the requirement:
- Each horizontal grid line (plate index `p`) is a distinct element/plate in the model.

Representation:
- For each plate index `p` that has any horizontal plate segment H[p,*] (or any node that hosts loads/supports), create/update one `jolt/model.Plate` entry.

Deriving Plate span and strip areas:
- Determine all active station indices for this plate:
  - active_nodes = { r | N[p,r] is true }
  - active_segments = { r | H[p,r] is true } where segment connects r -> r+1
- Compute:
  - first_row = min r where N[p,r] true OR H[p,r] touches r
  - last_row  = max r where N[p,r] true OR H[p,r] touches r
- Construct `A_strip` (length R-1):
  - For each r in 0..R-2:
    - if H[p,r] is true -> A_strip[r] = computed strip area based on plate properties (width*thickness or existing rule)
    - else -> A_strip[r] = 0 (or solver-accepted "inactive" value)
- Store plate properties (thickness/width/material/etc.) following existing rules.

Note: If the solver requires contiguous plate coverage, enforce contiguity:
- either disallow “holes” in H[p,*],
- or allow but warn & validate solver behavior.
Default decision: allow non-contiguous segments in UI but validate and show warning; solver-facing representation uses A_strip=0 for missing segments.

### Fasteners (fastener rows)
Interpretation aligned with the requirement:
- Each vertical grid station/column r that contains any fastener segment V[* ,r] corresponds to one fastener row entity in the model (conceptually “a fastener at station r”).

Representation:
- For each station r where any V[p,r] is true, create/update one `jolt/model.FastenerRow` entry anchored to that station index r.

Connections/topology:
- For each fastener segment V[p,r] between plates p and p+1:
  - Add a connection between plate p and plate p+1 at station r (exact encoding depends on existing FastenerRow schema).
- If the solver expects a list of connected plates at station r, derive it from contiguous V[* ,r] segments.

Countersink mapping:
- Requirement says countersunk is defined “at a node”.
- Implement as node-level metadata (p,r) and translate it to FastenerRow countersink/topology encoding at station r for the plate p (or plate side) as needed.

### Supports and point loads
Existing storage suggests tuples like `(plate_idx, local_node, value)`.
Mapping:
- plate_idx := p
- local_node := r (station index)
Thus:
- Support click at node (p,r) -> append/update `(p, r, support_value)`
- Load click at node (p,r) -> append/update `(p, r, load_value)`

---

## Persistence and round-trip fidelity (recommended)

Problem:
- JointConfiguration primitives may not preserve the exact click topology (explicit nodes, “empty” nodes, partial segments) in a lossless way.

Recommendation:
- Add an OPTIONAL UI metadata field to configuration, e.g. `JointConfiguration.ui_metadata` or `JointConfiguration.graphical_layout`:
  - version: integer
  - grid: {P, R}
  - nodes: list of (p,r)
  - plate_segments: list of (p,r) meaning H[p,r]
  - fastener_segments: list of (p,r) meaning V[p,r]
  - last_used_defaults: per entity type + per plate line/fastener station
  - pitch_prompt_shown: bool
  - last_pitch_value, last_pitch_index
This metadata must be ignored by solver and must not affect exports except preserving UI state.

Fallback:
- If graphical_layout is missing (older configs), reconstruct a best-effort grid from primitives:
  - nodes from plate spans + supports/loads anchors
  - H from A_strip>0
  - V from FastenerRow connections
This reconstruction should be deterministic and documented.

---

## Contextual parameter popovers & defaults

### Trigger rules
- On entity creation (plate segment / fastener segment / support / load):
  - show a small popover/modal to set required parameters.
- If the user already set defaults for that scope, popover is pre-filled and can be confirmed quickly.

### Defaults memory scopes
- Plate properties:
  - default per plate line (element p)
  - option “Apply to all segments in this element line”
- Fastener properties:
  - default per station r (fastener row)
  - plus global defaults for all fasteners
- Support/load:
  - global defaults + per-node override

### Editing after creation
- Clicking an existing entity in Select mode (or double-click) opens its properties popover.
- Model Parameters tab must allow full editing without using the canvas.

---

## Pitch UI, one-time prompt, and Shift+P hotkey

### Pitch inputs
- At the top boundary of the grid, between station r and r+1, provide a pitch input field.

### One-time prompt (first pitch entry)
- On the FIRST pitch value entry in a model:
  - show prompt: “Apply this pitch value to all remaining applicable positions to the right?”
  - include one-time hint: “Hotkey Shift+P fills remaining pitches.”
- Track that prompt was shown once per model via metadata/session state:
  - `pitch_prompt_shown = True`

### Applicable positions (deterministic definition)
Define:
- last_active_station = max r such that any node exists at station r across any plate line:
  - last_active_station = max { r | exists p: N[p,r] is true }
- last_applicable_pitch_index = max(0, last_active_station - 1)
Then “remaining applicable pitches” are pitch indices from current_pitch_index+1 to last_applicable_pitch_index.

### Shift+P behavior
- Hotkey Shift+P fills pitches to the right using the last entered pitch value:
  - start = last_edited_pitch_index
  - end = last_applicable_pitch_index
  - for i in start..end: pitches[i] = last_pitch_value
- Focus safety:
  - Do NOT trigger Shift+P while a text input is focused (pitch input or any form field).
  - Trigger only when Modeling canvas has focus (or when no input is focused).
- UX:
  - show a toast/status message confirming how many pitches were filled.

---

## Sidebar refactor & navigation

### Sidebar must contain only:
- Input Mode selector (only Standard)
- Method selector (Standard / Graphical)
- Dimension system selector
- Config loading (available configs)
- Available cases section + existing actions:
  - load, delete, export, export all cases, re-solve all cases
- Navigation links/buttons:
  - Model Definition
  - Model Comparison
  - Export/Reports

### Sidebar must NOT contain:
- Any geometry definition controls (rows/pitches/plates/fasteners/supports forms).
Those are moved into:
- Graphical: Modeling tab + Model Parameters tab
- Standard method: Model Parameters tab (form-based editor relocated from sidebar into content area)

---

## Technology approach (Streamlit realism)

The Modeling tab requires:
- clicking intersections and edges,
- hover feedback,
- popovers,
- keyboard hotkey Shift+P.

Recommendation:
- Implement Modeling grid as a custom Streamlit component (JS/React + SVG or Canvas) that emits events:
  - {type: 'click_node', p, r}
  - {type: 'click_h_edge', p, r}
  - {type: 'click_v_edge', p, r}
  - {type: 'key_shift_p'}
  - {type: 'hover', ...} (optional)
This is the most reliable way to implement edge hit-testing and keyboard shortcuts.

Fallback (if custom component is not feasible):
- Use a Plotly figure with click events for nodes/edges, but keyboard hotkeys will be harder and hover precision may be limited.
Document limitations if fallback is chosen.

---

## Migration and backward compatibility

- Saved configurations do not store input mode; existing files remain compatible.
- Normalize session state:
  - any legacy input_mode value coerces to "Standard" before rendering.
- Remove deprecated modes:
  - delete UI code paths for Refined Row and Node-based,
  - remove preprocessors that only serve those modes (unless still needed elsewhere).
- If graphical_layout metadata exists: use it to initialize Modeling grid.
- If not: reconstruct grid from existing primitives deterministically.

---

## PR plan (incremental, safe)

PR1: Discovery & cleanup
- Add/expand this doc.
- Remove deprecated input modes from UI & code paths.
- Add normalization/migration for legacy session values.

PR2: Sidebar + navigation refactor
- Move view navigation into sidebar.
- Remove geometry UI from sidebar without deleting underlying logic.

PR3: Modeling tab skeleton + grid component + node clicks
- Render grid, tool/mode toolbar.
- Implement create/delete node via mouse clicks + state sync.

PR4: Click-to-add/remove plate edges and fastener edges
- Implement horizontal edge (plate segment) toggling.
- Implement vertical edge (fastener segment) toggling.
- Implement supports/loads clicks on nodes.
- Validation + deletion cascades.

PR5: Model Parameters tab + contextual popovers + defaults
- Introduce Model Parameters/Results bottom tabs (Results unchanged).
- Implement property popovers on creation.
- Add bulk editing features (plates within element; global fastener compliance method; per-fastener override; global fastener params; node countersunk).

PR6: Pitch UI + one-time prompt + Shift+P + final tests
- Implement pitch inputs and propagation logic.
- Implement Shift+P hotkey in the modeling component.
- Add tests and polish.

---

## Test plan (practical)

Unit tests (pure Python):
- input_mode normalization (legacy -> Standard)
- grid->primitives mapping functions:
  - nodes/segments -> plates A_strip + spans
  - fastener segments -> FastenerRow connections
  - supports/loads mapping
- pitch algorithms:
  - last_active_station computation
  - one-time prompt gating
  - Shift+P fill range

Integration/UI tests:
- Minimal Streamlit integration tests (depending on existing repo tools):
  - smoke test: render Modeling tab without error
  - click event handling: simulate component outputs and verify state updates
- Optional e2e (Playwright/Selenium) if already used in repo.

---

## Assumptions & decisions (updated)
- Grid axes: vertical = plate lines (elements), horizontal = stations (fastener rows); pitches define station spacing.
- Node/edge toggling is explicit and click-first; edges require existing nodes at endpoints.
- UI metadata is allowed in config as solver-ignored data to preserve round-trip fidelity.
- Standard method forms move out of sidebar into the Model Parameters tab to satisfy “no geometry controls in sidebar” while preserving capability.
- “Each vertical line with fastener is a separate fastener” is interpreted as: each station r with any fastener segments corresponds to one FastenerRow entity anchored at r.
