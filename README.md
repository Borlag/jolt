# JOLT

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To start the application, run:

```bash
streamlit run jolt_app.py
```

## Selecting a fastener topology

Each `FastenerRow` can choose how its stiffness is connected into the 1D model.
Set `fastener.topology` (or place the same string in the JSON case file) to one of:

- `boeing_chain` – Canonical JOLT/Boeing 1D chain (no fastener DOF)
- `boeing_star_scaled` – Boeing double-shear stiffness with a scaled star
- `boeing_star_raw` – Legacy star using pairwise single-shear branches
- `empirical_chain` – Chain built from empirical (e.g., Huth) single-shear springs
- `empirical_star` – Empirical branches solved into a star (default for non-Boeing)

If `topology` is omitted, Boeing methods fall back to `boeing_star_raw` and other
methods use `empirical_star`. The tests in `tests/test_boeing_topologies.py` show
simple usage examples, including programmatic overrides.