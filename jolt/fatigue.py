"""
Fatigue analysis module for JOLT.
Implements Stress Severity Factor (SSF) analysis based on Niu Section 7.7 and Peterson's Stress Concentration Factors.
"""
from typing import Dict, Any, Union, List, Tuple
import math

# --- Embedded Data Tables ---
# Format: List of (x, y) tuples. Assumes sorted x.

# Ktb vs d/w (Bearing SCF)
# Source: Ktb.csv
DATA_KTB = [
    (0.0124, 1.0074), (0.0194, 1.0155), (0.0263, 1.0226), (0.0333, 1.0290), (0.0402, 1.0356),
    (0.0472, 1.0422), (0.0541, 1.0502), (0.0611, 1.0573), (0.0680, 1.0655), (0.0750, 1.0731),
    (0.0819, 1.0820), (0.0889, 1.0905), (0.0958, 1.0997), (0.1028, 1.1092), (0.1097, 1.1181),
    (0.1167, 1.1283), (0.1237, 1.1388), (0.1306, 1.1496), (0.1376, 1.1613), (0.1445, 1.1731),
    (0.1515, 1.1851), (0.1584, 1.2013), (0.1654, 1.2103), (0.1724, 1.2222), (0.1793, 1.2378),
    (0.1863, 1.2524), (0.1933, 1.2654), (0.2002, 1.2821), (0.2072, 1.2973), (0.2141, 1.3137),
    (0.2211, 1.3306), (0.2281, 1.3480), (0.2350, 1.3659), (0.2420, 1.3848), (0.2490, 1.4037),
    (0.2560, 1.4230), (0.2633, 1.4374), (0.2699, 1.4631), (0.2769, 1.4847), (0.2838, 1.5065),
    (0.2912, 1.5281), (0.2978, 1.5508), (0.3048, 1.5746), (0.3118, 1.5989), (0.3187, 1.6261),
    (0.3260, 1.6493), (0.3327, 1.6773), (0.3397, 1.7050), (0.3467, 1.7326), (0.3537, 1.7612),
    (0.3607, 1.7911), (0.3673, 1.8194), (0.3737, 1.8481), (0.3804, 1.8699), (0.3864, 1.9034),
    (0.3926, 1.9277), (0.3975, 1.9587), (0.4033, 1.9900), (0.4090, 2.0187), (0.4143, 2.0442),
    (0.4179, 2.0703), (0.4237, 2.0936), (0.4290, 2.1323), (0.4338, 2.1614), (0.4383, 2.1898),
    (0.4431, 2.2195), (0.4485, 2.2534), (0.4525, 2.2842), (0.4582, 2.3081), (0.4619, 2.3483),
    (0.4660, 2.3792), (0.4699, 2.4087), (0.4740, 2.4407), (0.4788, 2.4741), (0.4825, 2.5060),
    (0.4870, 2.5262), (0.4903, 2.5655), (0.4955, 2.5969), (0.4978, 2.6192), (0.5003, 2.6549),
    (0.5054, 2.6873), (0.5083, 2.7191), (0.5125, 2.7431), (0.5159, 2.7797), (0.5195, 2.8116),
    (0.5227, 2.8388), (0.5262, 2.8695), (0.5303, 2.9042), (0.5336, 2.9433), (0.5329, 2.9177),
    (0.5384, 2.9709), (0.5419, 3.0104), (0.5451, 3.0409), (0.5483, 3.0723), (0.5515, 3.1028),
    (0.5550, 3.1318), (0.5581, 3.1647), (0.5618, 3.1969), (0.5650, 3.2336), (0.5682, 3.2633),
    (0.5714, 3.2957), (0.5746, 3.3276), (0.5781, 3.3597), (0.5819, 3.3844), (0.5842, 3.4235),
    (0.5874, 3.4571), (0.5906, 3.4898), (0.5939, 3.5233), (0.5981, 3.5533), (0.6019, 3.5882),
    (0.6046, 3.6059), (0.6061, 3.6479), (0.6093, 3.6790), (0.6115, 3.6973)
]

# Theta vs t/d (Single Shear)
# Source: Single_Shear.csv
DATA_THETA_SINGLE = [
    (0.20, 1.197), (0.30, 1.305), (0.40, 1.410), (0.50, 1.515), (0.60, 1.628),
    (0.70, 1.736), (0.80, 1.850), (0.90, 1.972), (1.00, 2.080), (1.10, 2.211),
    (1.20, 2.336), (1.30, 2.482), (1.40, 2.624), (1.50, 2.782), (1.60, 2.956),
    (1.70, 3.125), (1.80, 3.320), (1.90, 3.495), (2.00, 3.679), (2.10, 3.848),
    (2.20, 4.025), (2.30, 4.215), (2.40, 4.395), (2.50, 4.567), (2.60, 4.751),
    (2.70, 4.943), (2.80, 5.115), (2.90, 5.284), (3.00, 5.461), (3.10, 5.645),
    (3.20, 5.819), (3.30, 5.991)
]

# Theta vs t/d (Double Shear)
# Source: Double_shear.csv
DATA_THETA_DOUBLE = [
    (0.30, 1.063), (0.40, 1.078), (0.50, 1.095), (0.60, 1.116), (0.70, 1.136),
    (0.80, 1.159), (0.90, 1.177), (1.00, 1.197), (1.10, 1.226), (1.20, 1.247),
    (1.30, 1.273), (1.40, 1.302), (1.50, 1.325), (1.60, 1.352), (1.70, 1.381),
    (1.80, 1.410), (1.90, 1.436), (2.00, 1.468), (2.10, 1.494), (2.20, 1.523),
    (2.30, 1.561), (2.40, 1.590), (2.50, 1.619), (2.60, 1.654), (2.70, 1.684),
    (2.80, 1.713), (2.90, 1.748), (3.00, 1.783), (3.10, 1.809), (3.20, 1.847),
    (3.30, 1.887), (3.40, 1.922), (3.50, 1.957), (3.60, 1.995), (3.70, 2.019),
    (3.80, 2.059), (3.90, 2.097), (4.00, 2.132), (4.10, 2.161), (4.20, 2.199),
    (4.30, 2.237), (4.40, 2.272), (4.50, 2.307), (4.60, 2.342), (4.70, 2.371),
    (4.80, 2.406), (4.90, 2.441), (5.00, 2.470), (5.10, 2.505), (5.20, 2.534),
    (5.30, 2.578), (5.40, 2.610), (5.50, 2.645), (5.60, 2.677), (5.70, 2.709),
    (5.80, 2.744), (5.90, 2.776), (6.00, 2.808), (6.10, 2.840), (6.20, 2.872),
    (6.30, 2.910), (6.40, 2.945), (6.50, 2.983), (6.60, 3.003), (6.70, 3.041),
    (6.80, 3.076), (6.90, 3.105), (7.00, 3.140)
]

def interpolate_table(x: float, table: List[Tuple[float, float]]) -> float:
    """Linear interpolation for sorted table."""
    if not table:
        return 1.0
    
    # Check bounds
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
        
    # Binary search or linear scan (linear is fine for small tables)
    for i in range(len(table) - 1):
        x1, y1 = table[i]
        x2, y2 = table[i+1]
        if x1 <= x <= x2:
            ratio = (x - x1) / (x2 - x1) if (x2 - x1) > 1e-9 else 0.0
            return y1 + ratio * (y2 - y1)
            
    return table[-1][1]

def calc_countersink_factor(
    depth_ratio: float, 
    is_countersunk: bool = False,
    cs_angle: float = 100.0,
    h: float = 0.0,
    b: float = 0.0
) -> float:
    """
    Calculate Countersink Correction Factor (Kcs).
    Based on Peterson Eq 4.97: K_cs = (0.72 * (h - b) / h + 1) * K_straight
    Here we return the multiplier (0.72 * ... + 1).
    
    Args:
        depth_ratio: Ratio of countersink depth to plate thickness (t_cs / t).
                     Used if simple depth ratio model is active.
        is_countersunk: Whether the hole is countersunk.
        cs_angle: Countersink angle (degrees).
        h: Plate thickness.
        b: Straight shank length.
    """
    if not is_countersunk:
        return 1.0
        
    # If we have explicit dimensions (h and b), use Eq 4.97
    # h = thickness, b = straight shank length
    # depth = h - b
    # ratio = depth / h
    # If b is provided (straight shank), we use it.
    # Otherwise we use depth_ratio.
    
    ratio = depth_ratio
    if h > 1e-9:
        # If b is not provided explicitly but we have ratio, b = h * (1 - ratio)
        # The formula is 1 + 0.72 * (h-b)/h = 1 + 0.72 * (depth)/h = 1 + 0.72 * ratio
        pass
        
    if ratio < 0: ratio = 0.0
    if ratio > 1.0: ratio = 1.0
        
    return 1.0 + 0.72 * ratio

def calc_ktn_eccentric(D: float, W: float, offset: float) -> float:
    """
    Calculate Net Tension SCF (Ktn) for an eccentric hole.
    Based on Peterson's Chart 4.3.
    """
    if W <= 0 or D >= W:
        return 0.0 # Invalid geometry
    
    # Dimensions
    c = (W / 2.0) - abs(offset) # Distance to near edge
    e = (W / 2.0) + abs(offset) # Distance to far edge
    a = D / 2.0 # Radius
    
    if c <= a:
        return 0.0 # Hole touches edge
        
    # Parameters
    lam = c / e
    x = a / c
    
    # Coefficients from Peterson Chart 4.3 (Polynomial approx)
    C1 = 2.989 - 0.0064 * lam
    C2 = -2.872 + 0.095 * lam
    C3 = 2.348 + 0.196 * lam
    
    ktn = C1 + C2 * x + C3 * (x**2)
    
    if ktn < 1.0: ktn = 1.0
    return ktn

def calc_ktg(D: float, W: float, offset: float) -> float:
    """
    Calculate Gross Tension SCF (Ktg).
    Based on Peterson's Chart 4.4.
    """
    if W <= 0 or D >= W:
        return 0.0
        
    c = (W / 2.0) - abs(offset)
    e = (W / 2.0) + abs(offset)
    a = D / 2.0
    
    if c <= a: return 0.0
    
    lam = c / e
    x = a / c
    
    # Coefficients from Peterson Chart 4.4
    # Ktg = C1 + C2(a/c) + C3(a/c)^2 + C4(a/c)^3
    
    lam2 = lam * lam
    
    C1 = 2.9969 - 0.0090 * lam + 0.01338 * lam2
    C2 = 0.1217 + 0.5180 * lam - 0.5297 * lam2
    C3 = 0.5565 + 0.7215 * lam + 0.6153 * lam2
    C4 = 4.082 + 6.0146 * lam - 3.9815 * lam2
    
    ktg = C1 + C2 * x + C3 * (x**2) + C4 * (x**3)
    
    if ktg < 1.0: ktg = 1.0
    return ktg

def calc_ktb(D: float, W: float) -> float:
    """
    Calculate Bearing SCF (Ktb).
    Interpolated from Niu Fig 7.7.25 (Ktb.csv).
    """
    if W <= 0: return 0.0
    ratio = D / W
    return interpolate_table(ratio, DATA_KTB)

def calc_theta(t: float, D: float, shear_type: str = "single") -> float:
    """
    Calculate Bearing Distribution Factor (Theta).
    Interpolated from Niu (Single_Shear.csv / Double_shear.csv).
    """
    if D <= 0: return 1.0
    ratio = t / D
    
    if "double" in shear_type.lower():
        return interpolate_table(ratio, DATA_THETA_DOUBLE)
    else:
        return interpolate_table(ratio, DATA_THETA_SINGLE)

def calculate_ssf(
    load_transfer: float,
    load_bypass: float,
    D: float,
    W: float,
    t: float,
    offset: float = 0.0,
    is_countersunk: bool = False,
    cs_depth_ratio: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    shear_type: str = "single",
    cs_affects_bypass: bool = False,
    reference_load: Union[float, None] = None
) -> Dict[str, float]:
    """
    Calculate Stress Severity Factor (SSF) with geometric corrections.
    """
    # Safeguards
    if W <= 0 or t <= 0 or D <= 0:
        return {
            "ssf": 0.0, "ktg": 0.0, "ktn": 0.0, "ktb": 0.0, "theta": 0.0,
            "sigma_ref": 0.0, "term_bearing": 0.0, "term_bypass": 0.0,
            "k_cs": 1.0, "t_eff": t
        }

    # 1. Geometry Adjustments
    # Effective bearing thickness
    t_eff = t
    if is_countersunk:
        t_eff = t * (1.0 - cs_depth_ratio)
    
    if t_eff <= 0: t_eff = 1e-6

    # Countersink Factor
    k_cs = calc_countersink_factor(cs_depth_ratio, is_countersunk)
    
    # 2. SCF Calculations
    # Ktn (Eccentric)
    ktn = calc_ktn_eccentric(D, W, offset)
    
    # Ktg (Gross)
    ktg_base = calc_ktg(D, W, offset)
    
    # Apply countersink penalty
    ktg = ktg_base * k_cs
    ktn = ktn * k_cs 
    
    # Ktb (Bearing)
    ktb = calc_ktb(D, W)
    
    # Theta
    theta = calc_theta(t, D, shear_type)
    
    # Bypass Area Adjustment
    area_gross = W * t
    if cs_affects_bypass and is_countersunk:
        # Approximate lost area due to CS
        extra_void = cs_depth_ratio * t * (cs_depth_ratio * t) # Rough approx
        area_gross -= extra_void
        
    if area_gross <= 0: area_gross = 1e-6
    
    if reference_load is not None:
        sigma_ref = reference_load / area_gross
    else:
        sigma_ref = (load_transfer + load_bypass) / area_gross
    
    if abs(sigma_ref) < 1e-9:
        return {
            "ssf": 0.0, "ktg": ktg, "ktn": ktn, "ktb": ktb, "theta": theta,
            "sigma_ref": sigma_ref, "term_bearing": 0.0, "term_bypass": 0.0,
            "k_cs": k_cs, "t_eff": t_eff
        }

    # Peak Stresses
    # term_bearing = Ktb * theta * (load_transfer / (D * t_eff))
    term_bearing = ktb * theta * (load_transfer / (D * t_eff))
    
    # term_bypass = Ktg * (load_bypass / Area_Gross)
    term_bypass = ktg * (load_bypass / area_gross)
    
    # 4. SSF
    # SSF = (alpha * beta / sigma_ref) * (term_bearing + term_bypass)
    ssf = (alpha * beta / sigma_ref) * (term_bearing + term_bypass)
    
    return {
        "ssf": ssf,
        "ktg": ktg,
        "ktn": ktn,
        "ktb": ktb,
        "theta": theta,
        "sigma_ref": sigma_ref,
        "term_bearing": term_bearing,
        "term_bypass": term_bypass,
        "k_cs": k_cs,
        "t_eff": t_eff
    }
