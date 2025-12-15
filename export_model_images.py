"""
Export scheme and load images for all verification test models.
"""
import json
from pathlib import Path

from jolt import Joint1D, Plate, FastenerRow, JointSolution
from jolt.config import JointConfiguration

try:
    from jolt.visualization_plotly import render_joint_diagram_plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not available, cannot generate images")


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_model_images(model_id: str, config_data: dict, output_dir: Path):
    """Export scheme and load images for a single model."""
    print(f"\nProcessing {model_id}...")
    
    # Build configuration
    config = JointConfiguration.from_dict(config_data)
    
    # Build and solve the model
    model = config.build_model()
    solution = model.solve(
        supports=config.supports,
        point_forces=config.point_forces if config.point_forces else None,
    )
    
    # Unit labels
    units = {
        "force": "lb",
        "stiffness": "lb/in",
        "displacement": "in",
        "stress": "psi",
        "length": "in",
    }
    
    # Export modes
    modes = ["scheme", "loads", "displacements"]
    
    for mode in modes:
        fig = render_joint_diagram_plotly(
            pitches=config.pitches,
            plates=config.plates,
            fasteners=config.fasteners,
            supports=config.supports,
            solution=solution,
            units=units,
            mode=mode,
            font_size=12,
        )
        
        if fig:
            # Try PNG first, fall back to HTML
            try:
                output_path = output_dir / f"{model_id}_{mode}.png"
                fig.write_image(str(output_path), width=1200, height=600, scale=2)
                print(f"  Saved: {output_path.name}")
            except Exception as e:
                # Fall back to HTML
                output_path = output_dir / f"{model_id}_{mode}.html"
                fig.write_html(str(output_path), include_plotlyjs='cdn')
                print(f"  Saved: {output_path.name} (HTML)")
        else:
            print(f"  Failed to generate {mode} figure")


def main():
    if not HAS_PLOTLY:
        print("Cannot export images without Plotly")
        return
    
    # Find test_values directory
    test_dir = Path("test_values")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    # Create output directory
    output_dir = Path("reports/model_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Find all config files
    config_files = sorted(test_dir.glob("*_config.json"))
    print(f"Found {len(config_files)} config files")
    
    for config_path in config_files:
        model_id = config_path.stem.replace("_config", "")
        config_data = load_config(config_path)
        
        try:
            export_model_images(model_id, config_data, output_dir)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Done! Images saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
