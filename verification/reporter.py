"""
Excel Report Generator for Verification Module.

Generates comprehensive Excel workbooks with:
- Comparison tables (Our values vs Reference vs Errors)
- Charts (fastener loads, displacements, axial forces)
- Conditional formatting for tolerances
- Summary statistics
"""

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .comparator import (
    ModelComparison, 
    CategoryComparison, 
    FieldComparison,
    get_tolerance,
)
from .runner import SolverResults, TestResult
from .loader import TestCase


# Constants
IMG_WIDTH_PX = 800
IMG_HEIGHT_PX = 400
IMG_SCALE = 2
DISPLAY_SCALE = 0.5


class VerificationReporter:
    """
    Generates Excel verification reports.
    
    Usage:
        reporter = VerificationReporter()
        reporter.add_model_comparison(comparison, solver_result, reference_data)
        reporter.save("reports/verification.xlsx")
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize the reporter.
        
        Args:
            output_path: Path to save the Excel file (optional, can set later)
        """
        if not HAS_XLSXWRITER:
            raise ImportError("xlsxwriter is required for Excel reports. Install with: pip install xlsxwriter")
            
        self.output_path = Path(output_path) if output_path else None
        self.comparisons: List[Tuple[ModelComparison, SolverResults, Dict]] = []
        
    def add_model_comparison(
        self,
        comparison: ModelComparison,
        solver_result: SolverResults,
        reference_data: Dict[str, Any],
    ):
        """
        Add a model comparison to the report.
        
        Args:
            comparison: ModelComparison results
            solver_result: Solver output data
            reference_data: Reference data for this formula
        """
        self.comparisons.append((comparison, solver_result, reference_data))
        
    def generate(self, output_path: Optional[str] = None) -> bytes:
        """
        Generate the Excel workbook.
        
        Args:
            output_path: Optional path to save file (overrides constructor path)
            
        Returns:
            Excel file as bytes
        """
        if output_path:
            self.output_path = Path(output_path)
            
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Create formats
        formats = self._create_formats(workbook)
        
        # Generate summary sheet
        self._write_summary_sheet(workbook, formats)
        
        # Generate sheet for each model/formula combination
        for comparison, solver_result, reference_data in self.comparisons:
            self._write_model_sheet(workbook, formats, comparison, solver_result, reference_data)
            
        workbook.close()
        output.seek(0)
        
        # Save to file if path specified
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'wb') as f:
                f.write(output.getvalue())
            output.seek(0)
            
        return output.getvalue()
    
    def save(self, output_path: str):
        """
        Generate and save the Excel report.
        
        Args:
            output_path: Path to save the file
        """
        self.generate(output_path)
        
    def _create_formats(self, workbook) -> Dict[str, Any]:
        """Create all cell formats for the workbook."""
        return {
            # Headers
            "header_main": workbook.add_format({
                'bold': True, 'font_size': 14, 'bg_color': '#1F4E79',
                'font_color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'
            }),
            "header": workbook.add_format({
                'bold': True, 'bg_color': '#D9E1F2', 'border': 1,
                'align': 'center', 'valign': 'vcenter', 'text_wrap': True
            }),
            "header_pass": workbook.add_format({
                'bold': True, 'bg_color': '#C6EFCE', 'font_color': '#006100',
                'border': 1, 'align': 'center'
            }),
            "header_fail": workbook.add_format({
                'bold': True, 'bg_color': '#FFC7CE', 'font_color': '#9C0006',
                'border': 1, 'align': 'center'
            }),
            
            # Numbers
            "num_3dec": workbook.add_format({'num_format': '0.000', 'border': 1}),
            "num_1dec": workbook.add_format({'num_format': '0.0', 'border': 1}),
            "num_6dec": workbook.add_format({'num_format': '0.000000', 'border': 1}),
            "sci": workbook.add_format({'num_format': '0.00E+00', 'border': 1}),
            "pct": workbook.add_format({'num_format': '0.00%', 'border': 1}),
            "pct_rel": workbook.add_format({'num_format': '0.00"%"', 'border': 1}),
            
            # Error highlighting
            "error_good": workbook.add_format({
                'num_format': '0.00"%"', 'border': 1, 
                'bg_color': '#C6EFCE', 'font_color': '#006100'
            }),
            "error_warn": workbook.add_format({
                'num_format': '0.00"%"', 'border': 1,
                'bg_color': '#FFEB9C', 'font_color': '#9C6500'
            }),
            "error_bad": workbook.add_format({
                'num_format': '0.00"%"', 'border': 1,
                'bg_color': '#FFC7CE', 'font_color': '#9C0006'
            }),
            
            # Text
            "text": workbook.add_format({'border': 1, 'align': 'left'}),
            "text_center": workbook.add_format({'border': 1, 'align': 'center'}),
            "text_bold": workbook.add_format({'bold': True, 'border': 1}),
        }
        
    def _write_summary_sheet(self, workbook, formats: Dict):
        """Write the summary sheet with overall results."""
        ws = workbook.add_worksheet("Summary")
        
        # Title
        ws.merge_range(0, 0, 0, 7, "JOLT Verification Report", formats["header_main"])
        ws.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        ws.write(2, 0, f"Total Models: {len(self.comparisons)}")
        
        # Summary table
        headers = ["Model", "Formula", "Status", "Max Rel Error (%)", 
                   "Fasteners Matched", "Nodes Matched", "Overall"]
        
        row = 4
        for col, header in enumerate(headers):
            ws.write(row, col, header, formats["header"])
            
        row += 1
        for comparison, solver_result, _ in self.comparisons:
            ws.write(row, 0, comparison.model_id, formats["text"])
            ws.write(row, 1, comparison.formula, formats["text_center"])
            
            status = "PASS" if comparison.overall_pass else "FAIL"
            status_fmt = formats["header_pass"] if comparison.overall_pass else formats["header_fail"]
            ws.write(row, 2, status, status_fmt)
            
            ws.write(row, 3, comparison.overall_max_rel_error, formats["num_3dec"])
            
            # Matched counts
            fast_cat = comparison.categories.get("fasteners")
            node_cat = comparison.categories.get("nodes")
            ws.write(row, 4, fast_cat.matched_count if fast_cat else 0, formats["text_center"])
            ws.write(row, 5, node_cat.matched_count if node_cat else 0, formats["text_center"])
            
            ws.write(row, 6, "✓" if comparison.overall_pass else "✗", status_fmt)
            row += 1
            
        # Auto-fit columns
        ws.set_column(0, 0, 20)
        ws.set_column(1, 1, 12)
        ws.set_column(2, 2, 10)
        ws.set_column(3, 3, 18)
        ws.set_column(4, 5, 16)
        ws.set_column(6, 6, 10)
        
    def _write_model_sheet(
        self, 
        workbook, 
        formats: Dict,
        comparison: ModelComparison,
        solver_result: SolverResults,
        reference_data: Dict,
    ):
        """Write a sheet for a single model/formula comparison."""
        # Sanitize sheet name
        sheet_name = f"{comparison.model_id}_{comparison.formula}"[:31]
        sheet_name = sheet_name.replace("/", "-").replace("\\", "-")
        
        try:
            ws = workbook.add_worksheet(sheet_name)
        except Exception:
            ws = workbook.add_worksheet()
            
        # Header
        ws.merge_range(0, 0, 0, 8, f"Verification: {comparison.model_id} ({comparison.formula})", 
                      formats["header_main"])
        ws.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        status = "PASS" if comparison.overall_pass else "FAIL"
        ws.write(2, 0, f"Status: {status}", 
                formats["header_pass"] if comparison.overall_pass else formats["header_fail"])
        ws.write(2, 2, f"Max Rel Error: {comparison.overall_max_rel_error:.3f}%")
        
        curr_row = 5
        
        # Write each category comparison
        for category_name in ["fasteners", "nodes", "plates", "loads"]:
            if category_name in comparison.categories:
                category = comparison.categories[category_name]
                curr_row = self._write_category_table(
                    ws, formats, curr_row, category_name, category,
                    solver_result, reference_data.get(category_name, [])
                )
                curr_row += 2
                
        # Write charts (if plotly available)
        if HAS_PLOTLY:
            curr_row = self._write_charts(
                ws, formats, curr_row, comparison, solver_result, reference_data
            )
            
    def _write_category_table(
        self,
        ws,
        formats: Dict,
        start_row: int,
        category_name: str,
        category: CategoryComparison,
        solver_result: SolverResults,
        reference_list: List[Dict],
    ) -> int:
        """Write a comparison table for a category."""
        title = category_name.title()
        
        # Section header
        ws.merge_range(start_row, 0, start_row, 10, 
                      f"{title} Comparison (Matched: {category.matched_count})", 
                      formats["header_main"])
        
        if category.matched_count == 0:
            ws.write(start_row + 1, 0, "No matching elements found", formats["text"])
            return start_row + 3
            
        row = start_row + 1
        
        # Build comparison data table
        # Headers: Key | Field | Model | Reference | Abs Error | Rel Error (%)
        headers = ["Element", "Field", "Model Value", "Reference", "Abs Error", "Rel Error (%)"]
        for col, header in enumerate(headers):
            ws.write(row, col, header, formats["header"])
        row += 1
        
        # Write data for each field
        for field_name, field_comp in category.fields.items():
            for i, key in enumerate(category.element_keys):
                if i >= len(field_comp.model_values):
                    continue
                    
                ws.write(row, 0, key, formats["text"])
                ws.write(row, 1, field_name, formats["text"])
                ws.write(row, 2, field_comp.model_values[i], formats["num_3dec"])
                ws.write(row, 3, field_comp.ref_values[i], formats["num_3dec"])
                ws.write(row, 4, field_comp.abs_errors[i], formats["num_3dec"])
                
                # Color-code relative error
                # Always show 100% errors as red (critical threshold)
                rel_err = field_comp.rel_errors[i]
                tolerance = get_tolerance(field_name)
                if rel_err >= 99.0:  # Critical threshold - always red
                    fmt = formats["error_bad"]
                elif rel_err <= tolerance * 0.5:
                    fmt = formats["error_good"]
                elif rel_err <= tolerance:
                    fmt = formats["error_warn"]
                else:
                    fmt = formats["error_bad"]
                ws.write(row, 5, rel_err, fmt)
                
                row += 1
                
        # Summary row
        row += 1
        ws.write(row, 0, "Summary Statistics", formats["text_bold"])
        row += 1
        
        summary_headers = ["Field", "Max Abs", "Mean Abs", "Max Rel (%)", "Mean Rel (%)", "RMS"]
        for col, header in enumerate(summary_headers):
            ws.write(row, col, header, formats["header"])
        row += 1
        
        for field_name, field_comp in category.fields.items():
            ws.write(row, 0, field_name, formats["text"])
            ws.write(row, 1, field_comp.max_abs, formats["num_3dec"])
            ws.write(row, 2, field_comp.mean_abs, formats["num_3dec"])
            ws.write(row, 3, field_comp.max_rel, formats["num_3dec"])
            ws.write(row, 4, field_comp.mean_rel, formats["num_3dec"])
            ws.write(row, 5, field_comp.rms, formats["num_6dec"])
            row += 1
            
        # Report unmatched elements
        if category.unmatched_model or category.unmatched_ref:
            row += 1
            ws.write(row, 0, "Unmatched Elements", formats["text_bold"])
            row += 1
            if category.unmatched_model:
                ws.write(row, 0, f"In model only: {', '.join(category.unmatched_model[:5])}", formats["text"])
                row += 1
            if category.unmatched_ref:
                ws.write(row, 0, f"In reference only: {', '.join(category.unmatched_ref[:5])}", formats["text"])
                row += 1
                
        # Set column widths
        ws.set_column(0, 0, 25)
        ws.set_column(1, 1, 15)
        ws.set_column(2, 5, 15)
        
        return row
        
    def _write_charts(
        self,
        ws,
        formats: Dict,
        start_row: int,
        comparison: ModelComparison,
        solver_result: SolverResults,
        reference_data: Dict,
    ) -> int:
        """Write comparison charts."""
        row = start_row
        
        # Fastener Load Comparison Chart
        if "fasteners" in comparison.categories:
            fast_cat = comparison.categories["fasteners"]
            if "force" in fast_cat.fields:
                fig = self._create_comparison_chart(
                    title=f"Fastener Load Comparison - {comparison.model_id}",
                    x_labels=fast_cat.element_keys,
                    model_values=fast_cat.fields["force"].model_values,
                    ref_values=fast_cat.fields["force"].ref_values,
                    y_label="Load (lb)"
                )
                row = self._insert_chart(ws, row, 0, "Fastener Loads", fig)
                
        # Node Displacement Comparison
        if "nodes" in comparison.categories:
            node_cat = comparison.categories["nodes"]
            if "displacement" in node_cat.fields:
                fig = self._create_comparison_chart(
                    title=f"Node Displacement Comparison - {comparison.model_id}",
                    x_labels=node_cat.element_keys,
                    model_values=node_cat.fields["displacement"].model_values,
                    ref_values=node_cat.fields["displacement"].ref_values,
                    y_label="Displacement (in)"
                )
                row = self._insert_chart(ws, row, 0, "Displacements", fig)
                
        return row
        
    def _create_comparison_chart(
        self,
        title: str,
        x_labels: List[str],
        model_values: List[float],
        ref_values: List[float],
        y_label: str,
    ) -> Optional["go.Figure"]:
        """Create a grouped bar chart comparing model vs reference."""
        if not HAS_PLOTLY:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Model',
            x=x_labels,
            y=model_values,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='Reference',
            x=x_labels,
            y=ref_values,
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Element',
            yaxis_title=y_label,
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=50, t=50, b=50),
        )
        
        return fig
        
    def _insert_chart(self, ws, row: int, col: int, caption: str, fig) -> int:
        """Insert a Plotly figure as an image."""
        if fig is None:
            return row
            
        try:
            img_bytes = fig.to_image(format="png", width=IMG_WIDTH_PX, height=IMG_HEIGHT_PX, scale=IMG_SCALE)
            image_data = io.BytesIO(img_bytes)
            
            ws.write(row, col, f"Figure: {caption}")
            ws.insert_image(row + 1, col, f'{caption}.png', {
                'image_data': image_data,
                'x_scale': DISPLAY_SCALE,
                'y_scale': DISPLAY_SCALE
            })
            
            # Calculate rows used
            display_height = IMG_HEIGHT_PX * DISPLAY_SCALE
            rows_needed = math.ceil(display_height / 15) + 3
            
            return row + rows_needed
            
        except Exception as e:
            ws.write(row, col, f"Error generating chart: {e}")
            return row + 2


def generate_markdown_summary(comparisons: List[ModelComparison]) -> str:
    """
    Generate a Markdown summary of verification results.
    
    Args:
        comparisons: List of ModelComparison objects
        
    Returns:
        Markdown formatted string
    """
    lines = [
        "# JOLT Verification Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Models:** {len(comparisons)}",
        "",
        "## Results",
        "",
        "| Model | Formula | Status | Max Rel Error (%) |",
        "|-------|---------|--------|-------------------|",
    ]
    
    passed = 0
    for comp in comparisons:
        status = "✅ PASS" if comp.overall_pass else "❌ FAIL"
        if comp.overall_pass:
            passed += 1
        lines.append(f"| {comp.model_id} | {comp.formula} | {status} | {comp.overall_max_rel_error:.3f} |")
        
    lines.extend([
        "",
        f"**Overall:** {passed}/{len(comparisons)} passed",
    ])
    
    return "\n".join(lines)


__all__ = [
    "VerificationReporter",
    "generate_markdown_summary",
]
