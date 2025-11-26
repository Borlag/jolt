import json
import json
from jolt import JointConfiguration, Joint1D

def verify_classic_results():
    with open("case_6_2.json", "r") as f:
        data = json.load(f)
    
    config = JointConfiguration.from_dict(data)
    
    # Reconstruct the model manually since apply_configuration updates session state which we don't have here
    # But we can use the logic from apply_configuration if we adapt it, or just build Joint1D directly.
    # config.to_model_inputs() doesn't exist, but we can extract from config.
    
    pitches = config.pitches
    plates = config.plates
    fasteners = config.fasteners
    supports = config.supports
    
    # Create model
    model = Joint1D(pitches, plates, fasteners)
    
    # Solve
    solution = model.solve(supports, point_forces=[])
    
    # Get Classic Results
    results = solution.classic_results_as_dicts()
    
    with open("verification_results.txt", "w") as f_out:
        f_out.write(f"{'Element':<10} {'Node':<6} {'Row':<4} {'L.Trans/P':<10} {'Detail':<8} {'Bearing':<8} {'Fbr/FDet':<8}\n")
        f_out.write("-" * 70 + "\n")
        for r in results:
            f_out.write(f"{r['Element']:<10} {r['Node']:<6} {r['Row']:<4} {r['L.Trans / P']:<10.3f} {r['Detail Stress']:<8.0f} {r['Bearing Stress']:<8.0f} {r['Fbr / FDetail']:<8.3f}\n")
    
    print("Verification results written to verification_results.txt")

if __name__ == "__main__":
    verify_classic_results()
