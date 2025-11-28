
import json
from jolt.model import Joint1D, Plate, FastenerRow, JointSolution
from jolt.inputs import process_node_based
import jolt.fatigue as fatigue

# Configuration from user
config_json = """
{
  "label": "Case_6_2_elements_row_D",
  "unloading": "",
  "pitches": [
    1.125,
    1.125,
    1.125,
    1.125,
    1.125,
    1.125,
    1.125,
    1.125,
    1.125
  ],
  "plates": [
    {
      "name": "Tripler",
      "E": 10000000.0,
      "t": 0.071,
      "first_row": 2,
      "last_row": 8,
      "A_strip": [
        0.067,
        0.067,
        0.067,
        0.067,
        0.067,
        0.067
      ],
      "Fx_left": 0.0,
      "Fx_right": 0.0
    },
    {
      "name": "Doubler",
      "E": 10000000.0,
      "t": 0.063,
      "first_row": 1,
      "last_row": 9,
      "A_strip": [
        0.059,
        0.059,
        0.059,
        0.059,
        0.059,
        0.059,
        0.059,
        0.059
      ],
      "Fx_left": 0.0,
      "Fx_right": 1000.0
    }
  ],
  "fasteners": [
    {
      "row": 2,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 3,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 4,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 5,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 6,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 7,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    },
    {
      "row": 8,
      "D": 0.188,
      "Eb": 10300000.0,
      "nu_b": 0.3,
      "method": "Boeing69",
      "connections": [
        [
          0,
          1
        ]
      ]
    }
  ],
  "supports": [
    [
      1,
      0,
      0.0
    ]
  ]
}
"""

data = json.loads(config_json)

# Reconstruct objects
plates = [Plate(**p) for p in data["plates"]]
fasteners = [FastenerRow(**f) for f in data["fasteners"]]
pitches = data["pitches"]
supports = [tuple(s) for s in data["supports"]]

# Solve
model = Joint1D(pitches, plates, fasteners)
solution = model.solve(supports)

# Compute Fatigue
solution.compute_fatigue_factors(fasteners)

# Check Node 1008 (Tripler, Row 8)
print("--- Fatigue Results ---")
for res in solution.fatigue_results:
    if res.node_id == 1008 or res.row == 8 or res.node_id == 1002:
        print(f"Node {res.node_id} (Row {res.row}, {res.plate_name}): SSF={res.ssf:.2f}, Ktb={res.ktb:.2f}, Theta={res.theta:.2f}, SigmaRef={res.sigma_ref:.2f}")
        print(f"  Brg={res.bearing_load:.2f}, Byp={res.bypass_load:.2f}, TermBrg={res.term_bearing:.2f}, TermByp={res.term_bypass:.2f}")

# Check Force Balance
# Logic from classic_results_as_dicts or similar?
# The user said "force balancing gives an error".
# Let's check the reactions and applied forces.

print("\n--- Force Balance ---")
sum_forces = 0.0
for p in plates:
    sum_forces += p.Fx_left + p.Fx_right
    
sum_reactions = sum(r.reaction for r in solution.reactions)
print(f"Sum Applied: {sum_forces}")
print(f"Sum Reactions: {sum_reactions}")
print(f"Balance: {sum_forces + sum_reactions}")

