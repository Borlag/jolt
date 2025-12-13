"""Comprehensive test script to verify the Jarfall compliance fix on all D06 cases."""
import json
from pathlib import Path
from jolt.model import Joint1D, Plate, FastenerRow

TEST_VALUES_DIR = Path(__file__).parent / "test_values"

def test_case(case_name):
    config_path = TEST_VALUES_DIR / f'{case_name}_config.json'
    ref_path = TEST_VALUES_DIR / f'{case_name}_reference.json'

    if not config_path.exists():
        print(f"  Skipping {case_name} - config not found")
        return None
    if not ref_path.exists():
        print(f"  Skipping {case_name} - reference not found")
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
    
    try:
        sol = model.solve(supports=supports)
    except Exception as e:
        print(f"  ERROR solving {case_name}: {e}")
        return None

    # Compare with reference
    ref_fasteners = ref.get('formulas', {}).get('boeing', {}).get('fasteners', [])
    if not ref_fasteners:
        print(f"  No reference fasteners found for {case_name}")
        return None
    
    print(f'=== {case_name} Fastener Force Comparison ===')
    max_err = 0
    max_err_pct = 0
    details = []
    
    for rf in ref_fasteners:
        ref_val = rf['force']
        matched = False
        for sf in sol.fasteners:
            pi = model.plates[sf.plate_i].name
            pj = model.plates[sf.plate_j].name
            if sf.row == rf['row'] and pi == rf['plate_i'] and pj == rf['plate_j']:
                calc_val = abs(sf.force)
                err = abs(calc_val - ref_val)
                err_pct = (err / ref_val * 100) if ref_val > 0.1 else 0
                max_err = max(max_err, err)
                max_err_pct = max(max_err_pct, err_pct)
                details.append({
                    'row': rf['row'],
                    'pi': pi[:8],
                    'pj': pj[:8],
                    'ref': ref_val,
                    'calc': calc_val,
                    'err': err,
                    'err_pct': err_pct
                })
                matched = True
                break
        if not matched:
            print(f"  WARNING: No match for R{rf['row']} {rf['plate_i']}-{rf['plate_j']}")

    # Print results sorted by error
    details.sort(key=lambda x: -x['err'])
    for d in details[:5]:  # Show top 5 errors
        print(f"  R{d['row']} {d['pi']:8s}-{d['pj']:8s}: Ref={d['ref']:7.1f} Calc={d['calc']:7.1f} Err={d['err']:6.2f} lbf ({d['err_pct']:.1f}%)")
    
    if len(details) > 5:
        print(f"  ... and {len(details) - 5} more fasteners")

    print(f"\n  Max Absolute Error: {max_err:.2f} lbf")
    print(f"  Max Relative Error: {max_err_pct:.1f}%")
    
    if max_err < 1.0:
        print("  PASS: Absolute error < 1.0 lbf")
    elif max_err < 5.0:
        print("  WARNING: Error between 1-5 lbf")
    else:
        print("  FAIL: Error > 5 lbf")
    
    return {'max_err': max_err, 'max_err_pct': max_err_pct, 'n_fasteners': len(details)}

def main():
    print("=" * 60)
    print("JARFALL COMPLIANCE FIX - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Find all config files
    cases = []
    for config_file in sorted(TEST_VALUES_DIR.glob("*_config.json")):
        case_name = config_file.stem.replace("_config", "")
        cases.append(case_name)
    
    print(f"\nFound {len(cases)} test cases: {cases}\n")
    
    results = {}
    for case in cases:
        print()
        result = test_case(case)
        if result is not None:
            results[case] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Case':<10} {'Fasteners':<12} {'Max Err (lbf)':<15} {'Max Err %':<12} {'Status':<8}")
    print("-" * 60)
    
    all_pass = True
    for case, r in results.items():
        status = "PASS" if r['max_err'] < 1.0 else ("WARN" if r['max_err'] < 5.0 else "FAIL")
        if r['max_err'] >= 1.0:
            all_pass = False
        print(f"{case:<10} {r['n_fasteners']:<12} {r['max_err']:<15.2f} {r['max_err_pct']:<12.1f} {status:<8}")
    
    print("-" * 60)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS NEED ATTENTION")

if __name__ == "__main__":
    main()
