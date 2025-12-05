"""Analysis utilities for comparing topology variants.

This module provides helpers for running the solver with multiple
fastener topology configurations and comparing results.
"""
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import Joint1D, FastenerRow, JointSolution, Plate


# Standard Boeing topology variants for comparison
TOPOLOGY_VARIANTS_BOEING = [
    "boeing_chain",
    "boeing_chain_eq1",
    "boeing_chain_eq2",
    "boeing_star_scaled",
    "boeing_star_raw",
    "boeing_star_eq1",
    "boeing_star_eq2",
]

# Standard empirical (Huth/Grumman/etc.) topology variants
TOPOLOGY_VARIANTS_EMPIRICAL = [
    "empirical_chain",
    "empirical_star",
]


def solve_with_topologies(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Iterable[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    point_forces: Optional[Sequence[Tuple[int, int, float]]] = None,
    topology_variants: Optional[Iterable[str]] = None,
) -> Dict[str, JointSolution]:
    """
    Run the same joint definition for several fastener topologies.
    
    This helper is useful for comparing how different topology choices
    affect fastener load distribution. It does NOT mutate the original
    fastener objects.
    
    Args:
        pitches: List of pitch values for the joint
        plates: List of Plate objects defining the joint
        fasteners: Iterable of FastenerRow objects
        supports: List of support tuples (plate_index, local_node, value)
        point_forces: Optional list of force tuples
        topology_variants: List of topology names to test; defaults to TOPOLOGY_VARIANTS_BOEING
    
    Returns:
        Dict mapping topology_name -> JointSolution
    
    Example:
        >>> results = solve_with_topologies(
        ...     pitches, plates, fasteners, supports,
        ...     topology_variants=["boeing_chain", "boeing_star_raw"]
        ... )
        >>> for topo, sol in results.items():
        ...     print(f"{topo}: max load = {max(abs(f.force) for f in sol.fasteners)}")
    """
    if topology_variants is None:
        topology_variants = TOPOLOGY_VARIANTS_BOEING
    
    results: Dict[str, JointSolution] = {}
    base_fasteners = list(fasteners)
    
    for topo in topology_variants:
        # Use dataclasses.replace to create new fastener objects with updated topology
        # This ensures we don't mutate the original fastener objects
        topo_fasteners = [replace(f, topology=topo) for f in base_fasteners]
        
        # Create a fresh model for each topology
        model = Joint1D(
            pitches=list(pitches),
            plates=list(plates),
            fasteners=topo_fasteners
        )
        
        # Solve and store result
        sol = model.solve(supports=list(supports), point_forces=list(point_forces) if point_forces else None)
        results[topo] = sol
    
    return results


__all__ = [
    "TOPOLOGY_VARIANTS_BOEING",
    "TOPOLOGY_VARIANTS_EMPIRICAL",
    "solve_with_topologies",
]
