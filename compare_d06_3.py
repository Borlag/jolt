"""Compare D06_3 results between boeing_star_scaled and boeing_beam topologies."""
import json
from jolt.model import Joint1D, Plate, FastenerRow

# Load D06_3
with open('test_values/D06_3_config.json') as f:
    config = json.load(f)
with open('test_values/D06_3_reference.json') as f:
    ref = json.load(f)

ref_fasteners = ref['formulas']['boeing']['fasteners']
ref_map = {}
for rf in ref_fasteners:
    key = (rf['row'], rf['plate_i'], rf['plate_j'])
    ref_map[key] = rf['force']

def test_topology(topo_name):
    plates = [Plate(**p) for p in config['plates']]
    fasteners = []
    for f_data in config['fasteners']:
        conns = [tuple(c) for c in f_data.get('connections', [])] if 'connections' in f_data else None
        kwargs = {k:v for k,v in f_data.items() if k != 'connections'}
        kwargs['topology'] = topo_name
        fasteners.append(FastenerRow(connections=conns, **kwargs))

    model = Joint1D(config['pitches'], plates, fasteners)
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in config['supports']]
    sol = model.solve(supports)
    
    errors = []
    for f in sol.fasteners:
        pi_name = sol.plates[f.plate_i].name
        pj_name = sol.plates[f.plate_j].name
        key = (f.row, pi_name, pj_name)
        if key in ref_map:
            ref_force = ref_map[key]
            err = 100 * abs(abs(f.force) - ref_force) / ref_force if ref_force > 0.1 else 0
            errors.append(err)
    return errors

star_errors = test_topology('boeing_star_scaled')
beam_errors = test_topology('boeing_beam')

print('D06_3 TOPOLOGY COMPARISON')
print('=' * 40)
print(f'Metric                    Star     Beam')
print(f'Avg Error               {sum(star_errors)/len(star_errors):5.1f}%   {sum(beam_errors)/len(beam_errors):5.1f}%')
print(f'Max Error               {max(star_errors):5.1f}%   {max(beam_errors):5.1f}%')

star_over15 = len([e for e in star_errors if e >= 15])
beam_over15 = len([e for e in beam_errors if e >= 15])
star_under10 = len([e for e in star_errors if e < 10])
beam_under10 = len([e for e in beam_errors if e < 10])

print(f'Errors >= 15%           {star_over15:5d}      {beam_over15:5d}')
print(f'Errors < 10%            {star_under10:5d}      {beam_under10:5d}')
