"""
Export logic for JOLT 1D Joint application.
Handles generation of Excel reports with aerospace-standard formatting.
"""
import io
import math
import pandas as pd
from typing import List, Dict, Any, Optional
import xlsxwriter
from jolt import JointConfiguration, JointSolution, Plate, FastenerRow
from jolt.units import UnitConverter, UnitSystem
from jolt.visualization_plotly import render_joint_diagram_plotly

# Constants for image sizing
IMG_WIDTH_PX = 1000
IMG_HEIGHT_PX = 600
IMG_SCALE = 2
DISPLAY_SCALE = 0.6
# Excel default row height is usually 15 points (20 pixels). 
# We use a conservative estimate to ensure no overlap.
EXCEL_ROW_HEIGHT_PX = 14 
COL_WIDTH_PX = 64 

class JoltExcelExporter:
    def __init__(self, models: List[JointConfiguration], unit_system: str):
        self.models = models
        self.units = unit_system
        self.labels = UnitConverter.get_labels(unit_system)
        self.output = io.BytesIO()
        self.workbook = xlsxwriter.Workbook(self.output, {'in_memory': True})
        
        # --- Aerospace Standard Formats ---
        # Blue header style typical of engineering reports
        self.fmt_header = self.workbook.add_format({
            'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 
            'align': 'center', 'valign': 'vcenter', 'text_wrap': True
        })
        self.fmt_header_main = self.workbook.add_format({
            'bold': True, 'font_size': 12, 'bg_color': '#4472C4', 
            'font_color': 'white', 'border': 1
        })
        # Precision formats
        self.fmt_num_3dec = self.workbook.add_format({'num_format': '0.000', 'border': 1})
        self.fmt_num_1dec = self.workbook.add_format({'num_format': '0.0', 'border': 1})
        self.fmt_sci = self.workbook.add_format({'num_format': '0.00E+00', 'border': 1})
        self.fmt_text = self.workbook.add_format({'border': 1, 'align': 'left'})
        self.fmt_fsi_crit = self.workbook.add_format({
            'num_format': '0.00', 'border': 1, 'bg_color': '#FFC7CE', 'font_color': '#9C0006'
        })
        self.fmt_fsi = self.workbook.add_format({'num_format': '0.00', 'border': 1})
        self.fmt_pct = self.workbook.add_format({'num_format': '0.00%', 'border': 1})

    def close(self):
        self.workbook.close()
        self.output.seek(0)
        return self.output

    def _write_table(self, worksheet, start_row: int, title: str, df: pd.DataFrame):
        """Helper to write a Pandas DataFrame with specific formats and auto-fit columns."""
        if df.empty:
            return start_row
            
        # Write Title
        worksheet.merge_range(start_row, 0, start_row, len(df.columns)-1, title, self.fmt_header_main)
        row = start_row + 1
        
        # Calculate column widths
        col_widths = [len(str(col)) for col in df.columns]
        
        # Write Header
        for col_num, value in enumerate(df.columns):
            worksheet.write(row, col_num, value, self.fmt_header)
        
        # Freeze panes (1 row for title, 1 row for header)
        if start_row == 5: # Assuming first table starts at row 5
             worksheet.freeze_panes(row + 1, 0)
             
        # Enable Autofilter
        worksheet.autofilter(row, 0, row + len(df), len(df.columns) - 1)
        
        row += 1
        
        # Write Data
        for _, record in df.iterrows():
            for col_num, col_name in enumerate(df.columns):
                val = record[col_name]
                
                # Update max column width
                val_str = str(val)
                # Cap max width to avoid extremely wide columns
                curr_len = min(len(val_str), 50) 
                if curr_len > col_widths[col_num]:
                    col_widths[col_num] = curr_len

                # Determine format based on column name context
                cell_fmt = self.fmt_text
                if isinstance(val, (int, float)):
                    if "FSI" in col_name or "Factor" in col_name:
                         # Conditional formatting for FSI >= 1.0 (Critical)
                         if "FSI" in col_name and val >= 1.0:
                             cell_fmt = self.fmt_fsi_crit
                         else:
                             cell_fmt = self.fmt_fsi
                    elif "Stress" in col_name or "Modulus" in col_name or "E [" in col_name:
                        cell_fmt = self.fmt_sci
                    elif "Load" in col_name or "Force" in col_name or "Bearing" in col_name:
                        cell_fmt = self.fmt_num_1dec
                    elif "Thickness" in col_name or "Area" in col_name or "Diameter" in col_name or "Displacement" in col_name:
                        cell_fmt = self.fmt_num_3dec
                    elif "%" in col_name:
                        cell_fmt = self.fmt_pct
                    else:
                        cell_fmt = self.fmt_num_3dec
                
                worksheet.write(row, col_num, val, cell_fmt)
            row += 1
            
        # Apply column widths
        for i, width in enumerate(col_widths):
            # Add a little padding
            worksheet.set_column(i, i, width + 2)
            
        return row + 2 # Add spacing

    def _insert_image_with_caption(self, ws, row: int, col: int, title: str, fig, scale: float = DISPLAY_SCALE) -> int:
        """
        Inserts a Plotly figure as an image with a caption.
        Returns the number of rows occupied by the image + caption + padding.
        """
        if not fig:
            return 0
            
        try:
            img_bytes = fig.to_image(format="png", width=IMG_WIDTH_PX, height=IMG_HEIGHT_PX, scale=IMG_SCALE)
            image_data = io.BytesIO(img_bytes)
            
            # Write Caption
            ws.write(row, col, f"Figure - {title}", self.fmt_header_main)
            
            # Insert Image
            # Calculate offset for image to be below caption
            ws.insert_image(row + 1, col, f'{title}.png', {
                'image_data': image_data, 
                'x_scale': scale, 
                'y_scale': scale
            })
            
            # Calculate rows needed
            # Image height in pixels * scale
            display_height = IMG_HEIGHT_PX * scale
            
            # Calculate rows needed more conservatively
            # Using a slightly smaller row height divisor ensures we allocate MORE rows than strictly needed
            # 14px is very safe (standard is ~20px)
            rows_needed = math.ceil(display_height / EXCEL_ROW_HEIGHT_PX) + 5 # +1 caption, +4 padding
            
            return rows_needed
        except Exception as e:
            ws.write(row, col, f"Error generating image '{title}': {str(e)}")
            return 2

    def export_model(self, model: JointConfiguration, options: Dict[str, Any]):
        # Sanitize sheet name
        safe_label = str(model.label).strip()[:25]
        sheet_name = f"Case_{safe_label}".replace(" ", "_").replace("/", "-")
        try:
            ws = self.workbook.add_worksheet(sheet_name)
        except Exception:
            ws = self.workbook.add_worksheet() # Fallback for duplicates
        
        # --- 1. Metadata ---
        ws.write(0, 0, "JOLT Analysis Report", self.fmt_header_main)
        ws.write(1, 0, f"Case: {model.label}")
        ws.write(2, 0, f"Units: {model.units}")
        ws.write(3, 0, f"Desc: {model.description}")
        ws.write(4, 0, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        ws.set_column(0, 0, 25) # Widen first column
        ws.set_column(1, 15, 15) # Set default width
        
        curr_row = 6
        
        # --- 2. Element Table ---
        if options.get("include_elements", True) and model.plates:
            data = []
            for p in model.plates:
                # Calculate Length/Pitch if possible (approximate)
                length = sum(model.pitches[max(0, p.first_row-1):min(len(model.pitches), p.last_row-1)]) if model.pitches else 0.0
                
                data.append({
                    "Layer Name": p.name,
                    "Material": p.material_name or "N/A",
                    f"E [{self.labels['stress']}]": p.E,
                    f"Thickness [{self.labels['length']}]": p.t,
                    f"Net Area [{self.labels['area']}]": p.A_strip[0] if p.A_strip else 0.0,
                    f"Bypass Area [{self.labels['area']}]": p.A_strip[0] if p.A_strip else 0.0, # Assuming strip area is bypass for now
                    f"Length [{self.labels['length']}]": length,
                    f"Fatigue Str [{self.labels['stress']}]": p.fatigue_strength or 0.0
                })
            df_elem = pd.DataFrame(data)
            curr_row = self._write_table(ws, curr_row, "Element Properties", df_elem)

        # --- 3. Fastener Table ---
        if options.get("include_fasteners", True) and model.fasteners:
            data = []
            for f in model.fasteners:
                # Determine interfaces
                interfaces = []
                if f.connections:
                     for (top, bot) in f.connections:
                         if top < len(model.plates) and bot < len(model.plates):
                             interfaces.append(f"{model.plates[top].name}-{model.plates[bot].name}")
                
                data.append({
                    "Row": f.row,
                    "Name": f.name,
                    f"Dia [{self.labels['length']}]": f.D,
                    "Method": f.method,
                    f"Bolt E [{self.labels['stress']}]": f.Eb,
                    "Countersunk": "Yes" if f.is_countersunk else "No",
                    "Interfaces": ", ".join(interfaces)
                })
            df_fast = pd.DataFrame(data)
            curr_row = self._write_table(ws, curr_row, "Fastener Definition", df_fast)
            
        # --- 4. Nodes Table (Optional) ---
        if options.get("include_nodes", False) and model.results:
            sol = JointSolution.from_dict(model.results)
            data = [n.as_dict() for n in sol.nodes]
            df_nodes = pd.DataFrame(data)
            # Filter/Rename columns
            if not df_nodes.empty:
                cols_map = {
                    "legacy_id": "Node ID", "plate_name": "Plate", "row": "Row", 
                    "x": f"X [{self.labels['length']}]", "net_bypass": f"Net Bypass [{self.labels['force']}]"
                }
                cols = [c for c in cols_map.keys() if c in df_nodes.columns]
                df_nodes = df_nodes[cols].rename(columns=cols_map)
                curr_row = self._write_table(ws, curr_row, "Node Definitions", df_nodes)

        # --- 5. Supports Table (Optional) ---
        if options.get("include_supports", False) and model.supports:
             data = []
             for s_idx, (p_idx, l_node, val) in enumerate(model.supports):
                 p_name = model.plates[p_idx].name if p_idx < len(model.plates) else str(p_idx)
                 data.append({
                     "ID": f"S{s_idx}",
                     "Plate": p_name,
                     "Local Node": l_node,
                     f"Prescribed Disp [{self.labels['length']}]": val
                 })
             df_supp = pd.DataFrame(data)
             curr_row = self._write_table(ws, curr_row, "Supports", df_supp)

        # --- 6. Results ---
        if model.results:
            sol = JointSolution.from_dict(model.results)
            
            # Internal Loads
            if options.get("include_loads", True):
                bb_data = sol.bearing_bypass_as_dicts()
                df_bb = pd.DataFrame(bb_data).rename(columns={
                    "Bearing [lb]": f"Bearing [{self.labels['force']}]", 
                    "Bypass [lb]": f"Bypass [{self.labels['force']}]"
                })
                curr_row = self._write_table(ws, curr_row, "Internal Loads", df_bb)
            
            # Displacements (Optional)
            if options.get("include_displacements", False):
                 data = []
                 for n in sol.nodes:
                     data.append({
                         "Node ID": n.legacy_id,
                         "Plate": n.plate_name,
                         "Row": n.row,
                         f"Displacement [{self.labels['length']}]": n.displacement
                     })
                 df_disp = pd.DataFrame(data)
                 curr_row = self._write_table(ws, curr_row, "Nodal Displacements", df_disp)

            # Fatigue
            if options.get("include_fatigue", True) and sol.fatigue_results:
                f_data = sol.fatigue_results_as_dicts()
                df_fat = pd.DataFrame(f_data)
                cols_to_keep = ["node_id", "plate_name", "row", "ktg", "ktb", "ssf", "peak_stress", "fsi"]
                cols_final = [c for c in cols_to_keep if c in df_fat.columns]
                df_fat = df_fat[cols_final].rename(columns={
                    "peak_stress": f"Peak Stress [{self.labels['stress']}]",
                    "fsi": "FSI"
                })
                curr_row = self._write_table(ws, curr_row, "Fatigue Analysis", df_fat)

            # --- 7. Images ---
            if options.get("include_images", True):
                label_spacing = options.get("label_spacing", 1.0)
                
                diagrams = []
                if options.get("img_scheme", True): diagrams.append(("scheme", "Scheme"))
                if options.get("img_loads", True): diagrams.append(("loads", "Load Distribution"))
                if options.get("img_disp", False): diagrams.append(("displacements", "Displacements"))
                if options.get("img_fatigue", False): diagrams.append(("fatigue", "Fatigue Criticality"))
                
                # Ensure we are below any existing content
                if ws.dim_rowmax is not None:
                    curr_row = max(curr_row, ws.dim_rowmax + 2)

                for mode, title in diagrams:
                    fig = render_joint_diagram_plotly(
                        model.pitches, model.plates, model.fasteners, model.supports, sol, 
                        self.labels, mode=mode, font_size=int(10 * label_spacing)
                    )
                    
                    rows_used = self._insert_image_with_caption(
                        ws, curr_row, 0, f"{title} ({model.label})", fig, scale=DISPLAY_SCALE
                    )
                    curr_row += rows_used

    def export_comparison(self, selected_models: List[JointConfiguration]):
        ws = self.workbook.add_worksheet("Comparison")
        ws.write(0, 0, "Model Comparison Matrix", self.fmt_header_main)
        ws.write(1, 0, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        # 1. Summary Matrix
        data = []
        solutions = {}
        for m in selected_models:
            if not m.results: continue
            solutions[m.model_id] = JointSolution.from_dict(m.results)
            res = m.results.get("summary", {})
            data.append({
                "Model": m.label,
                "Max FSI": res.get("max_fsi_global", 0),
                f"Max Load [{self.labels['force']}]": res.get("max_fastener_load", 0),
                f"Max Bypass [{self.labels['force']}]": res.get("max_bypass", 0)
            })
            
        df_comp = pd.DataFrame(data)
        curr_row = self._write_table(ws, 3, "High Level Metrics", df_comp)
        
        # 2. Detailed Fastener Loads Comparison
        # Pivot table: Rows = Fastener Row, Cols = Model Label, Values = Load
        all_loads = []
        for m in selected_models:
            if m.model_id not in solutions: continue
            sol = solutions[m.model_id]
            for f in sol.fasteners:
                all_loads.append({
                    "Row": f.row,
                    "Model": m.label,
                    "Load": abs(f.force)
                })
        
        if all_loads:
            df_loads = pd.DataFrame(all_loads)
            df_pivot_loads = df_loads.pivot_table(index="Row", columns="Model", values="Load").reset_index()
            # Rename columns to include units
            df_pivot_loads.columns.name = None
            df_pivot_loads = df_pivot_loads.rename(columns={c: f"{c} [{self.labels['force']}]" for c in df_pivot_loads.columns if c != "Row"})
            curr_row = self._write_table(ws, curr_row, "Fastener Load Distribution", df_pivot_loads)
            
        # 3. Detailed FSI Comparison (Top Critical Nodes)
        all_fsi = []
        for m in selected_models:
            if m.model_id not in solutions: continue
            sol = solutions[m.model_id]
            if sol.fatigue_results:
                # Take top 5 critical nodes per model
                top_nodes = sorted(sol.fatigue_results, key=lambda x: x.fsi, reverse=True)[:5]
                for n in top_nodes:
                    all_fsi.append({
                        "Model": m.label,
                        "Node": f"{n.plate_name} @ {n.row}",
                        "FSI": n.fsi,
                        f"Peak Stress [{self.labels['stress']}]": n.peak_stress
                    })
        
        if all_fsi:
            df_fsi = pd.DataFrame(all_fsi)
            curr_row = self._write_table(ws, curr_row, "Top Critical Nodes (FSI)", df_fsi)
            
        # 4. Comparison Diagrams Grid
        # We will render diagrams for all models side-by-side
        # Grid: Rows = Diagram Type, Cols = Models
        
        # Calculate column spacing for images
        # Image width ~1000px * scale. Default col width ~64px.
        # Cols needed = (1000 * scale) / 64
        grid_scale = 0.5 # Slightly smaller for comparison
        cols_per_img = math.ceil((IMG_WIDTH_PX * grid_scale) / COL_WIDTH_PX) + 1
        
        if ws.dim_rowmax is not None:
            curr_row = max(curr_row, ws.dim_rowmax + 2)
            
        ws.write(curr_row, 0, "Visual Comparison", self.fmt_header_main)
        curr_row += 2
        
        diagram_types = [
            ("scheme", "Scheme Overview"),
            ("loads", "Load Distribution"),
            ("fatigue", "Fatigue Analysis")
        ]
        
        for mode, title in diagram_types:
            max_rows_in_strip = 0
            
            # Header for this strip
            ws.write(curr_row, 0, f"--- {title} ---", self.fmt_header)
            curr_row += 2
            
            for i, m in enumerate(selected_models):
                if m.model_id not in solutions: continue
                sol = solutions[m.model_id]
                
                fig = render_joint_diagram_plotly(
                    m.pitches, m.plates, m.fasteners, m.supports, sol, 
                    self.labels, mode=mode, font_size=10
                )
                
                col_idx = i * cols_per_img
                rows_used = self._insert_image_with_caption(
                    ws, curr_row, col_idx, f"{m.label}", fig, scale=grid_scale
                )
                
                max_rows_in_strip = max(max_rows_in_strip, rows_used)
            
            curr_row += max_rows_in_strip + 1 # Add spacing between strips
