"""Quick test script to verify the Jarfall compliance fix on D06_3 and D06_4."""
import json
from pathlib import Path
from jolt.model import Joint1D, Plate, FastenerRow

def test_case(case_name):
    config_path = Path(f'test_values/{case_name}_config.json')
    ref_path = Path(f'test_values/{case_name}_reference.json')

    if not config_path.exists() or not ref_path.exists():
        print(f"Skipping {case_name} - files not found")
        return None

    with open(config_path) as f:
        config = json.load(f)
    with open(ref_path) as f:
        ref = json.load(f)

    plates = [Plate(**p) for p in config['plates']]
    fasteners = []
    for f_data in config['fasteners']:
        conns = None
        if 'connections' in f_data:
            conns = [tuple(c) for c in f_data['connections']]
        kwargs = f_data.copy()
        if 'connections' in kwargs:
            del kwargs['connections']
        fasteners.append(FastenerRow(connections=conns, **kwargs))

    model = Joint1D(config['pitches'], plates, fasteners)
    supports = [tuple(s) for s in config['supports']]
    sol = model.solve(supports=supports)

    # Compare with reference
    ref_fasteners = ref['formulas']['boeing']['fasteners']
    print(f'=== {case_name} Fastener Force Comparison ===')
    max_err = 0
    for rf in ref_fasteners:
        ref_val = rf['force']
        for sf in sol.fasteners:
            pi = model.plates[sf.plate_i].name
            pj = model.plates[sf.plate_j].name
            if sf.row == rf['row'] and pi == rf['plate_i'] and pj == rf['plate_j']:
                calc_val = abs(sf.force)
                err = abs(calc_val - ref_val)
                max_err = max(max_err, err)
                print(f"R{rf['row']} {pi[:8]:8s}-{pj[:8]:8s}: Ref={ref_val:7.1f} Calc={calc_val:7.1f} Err={err:6.2f} lbf")
                break

    print(f"\nMax Absolute Error: {max_err:.2f} lbf")
    if max_err < 1.0:
        print("PASS: Error < 1.0 lbf")
    elif max_err < 5.0:
        print("WARNING: Error between 1-5 lbf")
    else:
        print("FAIL: Error > 5 lbf")
    return max_err

def main():
    results = {}
    for case in ["D06_3", "D06_4"]:
        print()
        err = test_case(case)
        if err is not None:
            results[case] = err
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for case, err in results.items():
        status = "PASS" if err < 1.0 else ("WARN" if err < 5.0 else "FAIL")
        print(f"{case}: Max Error = {err:.2f} lbf [{status}]")

if __name__ == "__main__":
    main()

