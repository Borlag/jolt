# JOLT Verification Module

Standalone automated test module for validating the JOLT joint solver against Boeing JOLT reference results.

## Overview

This module provides:
- **Automated testing** of solver outputs against Boeing JOLT reference values
- **Error statistics** with per-element and aggregate metrics
- **Excel reports** with comparison tables, charts, and conditional formatting
- **CLI interface** for easy integration into CI/CD pipelines

## Quick Start

```bash
# Run all tests
python run_verification.py

# Run with verbose output
python run_verification.py --verbose

# Run specific model
python run_verification.py --model D06

# List available test cases
python run_verification.py --list
```

## Directory Structure

```
jolt-1/
├── test_values/                    # Test data directory
│   ├── D06_config.json             # Model configuration
│   ├── D06_reference.json          # Reference results
│   └── ...
├── verification/                   # Verification module
│   ├── __init__.py
│   ├── schemas.py                  # JSON schema definitions
│   ├── loader.py                   # Test case loader
│   ├── runner.py                   # Solver execution
│   ├── comparator.py               # Error computation
│   ├── reporter.py                 # Excel report generator
│   └── cli.py                      # Command-line interface
├── run_verification.py             # Entry point script
└── reports/                        # Generated reports (auto-created)
    └── verification_report.xlsx
```

## JSON Schema Specifications

### Configuration File (`{model_id}_config.json`)

Standard JOLT configuration format:

```json
{
  "label": "Case Name",
  "pitches": [1.125, 1.125, ...],
  "plates": [
    {
      "name": "Doubler",
      "E": 10000000.0,
      "t": 0.071,
      "first_row": 2,
      "last_row": 5,
      "A_strip": [0.067, 0.067, 0.067],
      "Fx_left": 0.0,
      "Fx_right": 0.0
    }
  ],
  "fasteners": [
    {
      "row": 2,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [[0, 1], [1, 2]]
    }
  ],
  "supports": [[0, 3, 0.0]]
}
```

### Reference File (`{model_id}_reference.json`)

Reference results keyed by formula:

```json
{
  "model_id": "D06",
  "description": "Boeing JOLT reference results",
  "source": "Boeing JOLT v2.x",
  "formulas": {
    "boeing": {
      "fasteners": [
        {
          "row": 2,
          "plate_i": "Doubler",
          "plate_j": "Skin",
          "force": 364.8,
          "stiffness": 151779.46,
          "bearing_upper": 364.8,
          "bearing_lower": 364.8
        }
      ],
      "nodes": [
        {
          "plate_name": "Doubler",
          "row": 2,
          "displacement": 0.0,
          "net_bypass": 0.0
        }
      ],
      "plates": [
        {
          "plate_name": "Doubler",
          "segment": 0,
          "axial_force": 1000.0,
          "stiffness": 670000.0
        }
      ],
      "loads": [
        {
          "element": "Doubler",
          "row": 2,
          "incoming_load": 1000.0,
          "bypass_load": 635.2,
          "load_transfer": 364.8
        }
      ],
      "reactions": [
        {
          "plate_name": "Doubler",
          "local_node": 3,
          "reaction": 1000.0
        }
      ]
    },
    "huth": {
      // Same structure for Huth formula
    }
  }
}
```

### Units Convention

All values in reference files use Imperial units:

| Field | Unit |
|-------|------|
| Stiffness | lb/in |
| Displacement | in |
| Loads/Forces | lb |
| Stress | psi |
| Thickness | in |
| Area | in² |

## How to Add New Test Cases

1. **Create Configuration File**
   
   Copy your existing JOLT configuration to `test_values/{model_id}_config.json`
   
   ```bash
   copy your_config.json test_values/NewCase_config.json
   ```

2. **Create Reference File**
   
   Generate a sample template:
   ```bash
   python run_verification.py --create-sample NewCase
   ```
   
   This creates `test_values/NewCase_reference.json` with the required structure.

3. **Populate Reference Values**
   
   Edit the reference file with actual Boeing JOLT output values for each field.
   
   > **Note**: Only include fields you want to validate. Fields not present in the reference are skipped.

4. **Run Verification**
   
   ```bash
   python run_verification.py --model NewCase --verbose
   ```

## How to Add New Formulas

1. **Add Formula Alias** (if needed)
   
   Edit `verification/schemas.py` and add to `FORMULA_ALIASES`:
   
   ```python
   FORMULA_ALIASES = {
       # ...existing...
       "new_formula": "NewFormula_Method",
   }
   ```

2. **Add Reference Data**
   
   In your reference JSON, add a new key under `formulas`:
   
   ```json
   {
     "formulas": {
       "boeing": { ... },
       "new_formula": {
         "fasteners": [...],
         "nodes": [...],
         ...
       }
     }
   }
   ```

3. **Run Verification**
   
   ```bash
   python run_verification.py --formula new_formula
   ```

## CLI Usage

```
usage: run_verification.py [-h] [-m MODEL] [-f FORMULA] [-o OUTPUT] 
                           [-d TEST_DIR] [-v] [-l] [--create-sample MODEL_ID]
                           [--markdown]

Options:
  -m, --model MODEL        Model ID(s) to run (can specify multiple)
  -f, --formula FORMULA    Formula to use (boeing, huth, etc.)
  -o, --output PATH        Output Excel report path
  -d, --test-dir DIR       Directory containing test files
  -v, --verbose            Print detailed progress
  -l, --list               List available test cases
  --create-sample ID       Create sample reference file
  --markdown               Output Markdown summary to console
```

### Examples

```bash
# Run all tests with default settings
python run_verification.py

# Run only Boeing formula tests
python run_verification.py --formula boeing --verbose

# Run multiple specific models
python run_verification.py -m D06 -m CaseA -m CaseB

# Custom output location
python run_verification.py --output my_reports/validation_2024.xlsx

# Generate Markdown summary for CI logs
python run_verification.py --markdown
```

## Report Location

Generated Excel reports are saved to `reports/verification_report.xlsx` by default.

Each report contains:
- **Summary Sheet**: Overall pass/fail status for all models
- **Model Sheets**: Detailed comparison tables and charts per model/formula

## Default Tolerances

| Field Type | Absolute | Relative |
|------------|----------|----------|
| Force/Load | ±0.5 lb | ±0.5% |
| Stiffness | ±100 lb/in | ±1.0% |
| Displacement | ±1e-5 in | ±1.0% |
| Stress | ±10 psi | ±1.0% |
| Default | ±0.1 | ±1.0% |

## Integration with pytest

The verification module can be used with pytest:

```python
# tests/test_verification.py
import pytest
from verification import TestLoader, TestRunner, compare_results

def test_d06_boeing():
    loader = TestLoader("test_values")
    case = loader.load_single("D06")
    assert case is not None and case.is_valid
    
    runner = TestRunner()
    result = runner.run_case(case, "boeing")
    assert result.success
    
    ref_data = case.reference_data["formulas"]["boeing"]
    comparison = compare_results(
        "D06", "boeing",
        {"fasteners": result.fasteners},
        ref_data
    )
    
    assert comparison.overall_pass, f"Max error: {comparison.overall_max_rel_error}%"
```

## Troubleshooting

**No test cases found**
- Ensure files follow naming convention: `{model_id}_config.json` and `{model_id}_reference.json`
- Check the test directory path (default: `test_values/`)

**Model/reference mismatch**
- Verify `model_id` in reference JSON matches the filename prefix

**Missing xlsxwriter**
- Install with: `pip install xlsxwriter`

**Charts not appearing**
- Install Plotly and kaleido: `pip install plotly kaleido`
