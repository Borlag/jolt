"""Trace mixed-shear branch compliances for D06_3."""
import json
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    config_path = Path(__file__).parent / "test_values" / "D06_3_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Extract fastener params (row 2)
    fastener = config["fasteners"][0]
    D = fastener["D"]
    Eb = fastener["Eb"]
    nu_b = fastener.get("nu_b", 0.3)
    
    # Extract plates at row 2
    plates = []
    for p in config["plates"]:
        if p["first_row"] <= 2 <= p["last_row"]:
            plates.append(p)
    
    print(f"=== MIXED-SHEAR BRANCH COMPLIANCE TRACE ===")
    print(f"Plates: {[p['name'] for p in plates]}")
    n_branches = len(plates)
    print(f"n_branches = {n_branches}")
    
    # Compute branch compliances using mixed-shear model
    G_b = Eb / (2.0 * (1.0 + nu_b))
    A_b = math.pi * (D / 2.0) ** 2
    I_b = math.pi * (D / 2.0) ** 4 / 4.0
    
    print(f"\nFastener: D={D}, Eb={Eb}, nu_b={nu_b}")
    print(f"G_b={G_b:.0f}, A_b={A_b:.6f}, I_b={I_b:.9f}")
    
    print("\n--- BRANCH COMPLIANCES ---")
    branch_compliances = []
    for i, p in enumerate(plates):
        t = p["t"]
        E = p["E"]
        
        t_shear = min(t, (D + t) / 2.0)
        C_s = (4.0 * t_shear) / (9.0 * G_b * A_b)
        
        t_bend = min(t, D)
        C_b = (6.0 * t_bend**3) / (40.0 * Eb * I_b)
        
        C_brg_raw = (1.0 / t) * (1.0 / Eb + 1.0 / E)
        
        is_internal = (0 < i < n_branches - 1)
        C_brg = C_brg_raw / 2.0 if is_internal else C_brg_raw
        
        C_total = C_s + C_b + C_brg
        K_branch = 1.0 / C_total
        
        branch_compliances.append(C_total)
        
        label = "(internal - double shear)" if is_internal else "(external - single shear)"
        print(f"  Branch {i} ({p['name'][:10]:10}): C={C_total:.6e}  K={K_branch:.0f}  {label}")
    
    # Compute effective pairwise stiffnesses from branches
    print("\n--- EFFECTIVE PAIRWISE STIFFNESSES (from branches) ---")
    for i in range(n_branches - 1):
        C_eff = branch_compliances[i] + branch_compliances[i+1]
        K_eff = 1.0 / C_eff
        p_i = plates[i]
        p_j = plates[i+1]
        print(f"  {p_i['name'][:8]:8}-{p_j['name'][:8]:8}: C_eff={C_eff:.6e}  K_eff={K_eff:.0f}")
    
    # Compare with Boeing 69 single-shear pairwise
    from jolt.fasteners import boeing69_compliance
    print("\n--- BOEING 69 SINGLE-SHEAR PAIRWISE (for reference) ---")
    for i in range(n_branches - 1):
        p_i = plates[i]
        p_j = plates[i+1]
        C_pair = boeing69_compliance(
            ti=p_i["t"], Ei=p_i["E"],
            tj=p_j["t"], Ej=p_j["E"],
            Eb=Eb, nu_b=nu_b,
            diameter=D, shear_planes=1
        )
        K_pair = 1.0 / C_pair
        print(f"  {p_i['name'][:8]:8}-{p_j['name'][:8]:8}: C_pair={C_pair:.6e}  K_pair={K_pair:.0f}")


if __name__ == "__main__":
    main()
