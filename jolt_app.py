"""Streamlit application for the JOLT 1D joint model."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

try:  # Optional imports used for the UI only
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
except Exception:  # pragma: no cover - plotting is optional for tests
    plt = None  # type: ignore
    Line2D = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except Exception as exc:  # pragma: no cover - tests do not require Streamlit
    raise RuntimeError(
        "The Streamlit interface requires the 'streamlit' package. Install it to launch the app."
    ) from exc

from jolt import FastenerRow, Joint1D, JointSolution, Plate, figure76_example


st.set_page_config(page_title="JOLT 1D Joint", layout="wide")


def _node_table(solution: JointSolution) -> List[dict]:
    return solution.nodes_as_dicts()


def _bar_table(solution: JointSolution) -> List[dict]:
    return solution.bars_as_dicts()


def _bearing_table(solution: JointSolution) -> List[dict]:
    return solution.bearing_bypass_as_dicts()


# ---------- 2.1 Example configuration ----------

def load_example_figure76():
    return figure76_example()


# ---------- 2.2 Session state ----------

if "pitches" not in st.session_state:
    (
        st.session_state.pitches,
        st.session_state.plates,
        st.session_state.fasteners,
        st.session_state.supports,
    ) = load_example_figure76()


# ---------- 2.3 Input panels ----------

with st.sidebar:
    st.header("Geometry")
    n_rows = st.number_input("Number of rows", 1, 50, len(st.session_state.pitches))
    if n_rows != len(st.session_state.pitches):
        if n_rows > len(st.session_state.pitches):
            last = st.session_state.pitches[-1]
            st.session_state.pitches.extend([last] * (n_rows - len(st.session_state.pitches)))
        else:
            st.session_state.pitches = st.session_state.pitches[:n_rows]
    cols = st.columns(2)
    with cols[0]:
        same_pitch = st.checkbox("All pitches equal", value=True)
    if same_pitch:
        value = st.number_input("Pitch value [in]", 0.01, 100.0, st.session_state.pitches[0], step=0.001, format="%.3f")
        st.session_state.pitches = [float(value)] * n_rows
    else:
        st.write("Pitches [in]")
        st.session_state.pitches = [
            st.number_input(f"p[{i+1}]", 0.001, 100.0, st.session_state.pitches[i], key=f"pitch_{i}", step=0.001, format="%.3f")
            for i in range(n_rows)
        ]

    st.divider()
    st.subheader("Plates (Layers)")
    for idx, plate in enumerate(st.session_state.plates):
        with st.expander(f"Layer {idx}: {plate.name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            plate.name = c1.text_input("Name", plate.name, key=f"pl_name_{idx}")
            plate.E = c2.number_input("E [psi]", 1e5, 5e8, plate.E, key=f"pl_E_{idx}", step=1e5, format="%.0f")
            plate.t = c3.number_input("t [in]", 0.001, 2.0, plate.t, key=f"pl_t_{idx}", step=0.001, format="%.3f")
            d1, d2, _ = st.columns(3)
            plate.first_row = int(d1.number_input("First row", 1, n_rows, plate.first_row, key=f"pl_fr_{idx}"))
            plate.last_row = int(d2.number_input("Last row", 1, n_rows, plate.last_row, key=f"pl_lr_{idx}"))
            st.write(f"Segments = {plate.last_row - plate.first_row + 1}")
            segments = plate.last_row - plate.first_row + 1
            if len(plate.A_strip) != segments:
                default_area = plate.A_strip[0] if plate.A_strip else 0.05
                plate.A_strip = [default_area] * segments
            same_area = st.checkbox("Same bypass area for all segments", value=True, key=f"sameA_{idx}")
            if same_area:
                area_val = st.number_input(
                    "Bypass area per segment [in²]", 1e-5, 10.0, plate.A_strip[0], key=f"pl_A_all_{idx}", step=0.001, format="%.3f"
                )
                plate.A_strip = [float(area_val)] * segments
            else:
                for seg in range(segments):
                    plate.A_strip[seg] = st.number_input(
                        f"A[{seg+1}] [in²]", 1e-5, 10.0, plate.A_strip[seg], key=f"pl_A_{idx}_{seg}", step=0.001, format="%.3f"
                    )
            e1, e2 = st.columns(2)
            plate.Fx_left = e1.number_input(
                "End load LEFT [+→] [lb]", -1e6, 1e6, plate.Fx_left, key=f"pl_Fl_{idx}", step=1.0, format="%.1f"
            )
            plate.Fx_right = e2.number_input(
                "End load RIGHT [+→] [lb]", -1e6, 1e6, plate.Fx_right, key=f"pl_Fr_{idx}", step=1.0, format="%.1f"
            )
    cadd, cex = st.columns([1, 1])
    if cadd.button("➕ Add layer"):
        st.session_state.plates.append(
            Plate(
                name=f"Layer{len(st.session_state.plates)}",
                E=1.0e7,
                t=0.05,
                first_row=1,
                last_row=n_rows,
                A_strip=[0.05] * n_rows,
            )
        )
    if cex.button("🗑 Remove last layer") and len(st.session_state.plates) > 1:
        st.session_state.plates.pop()

    st.divider()
    st.subheader("Fasteners")
    fcols = st.columns([1, 3])
    with fcols[0]:
        if st.button("➕ Add fastener"):
            if st.session_state.fasteners:
                template = st.session_state.fasteners[-1]
            else:
                template = FastenerRow(row=1, D=0.188, Eb=1.0e7, nu_b=0.3)
            default_row = min(len(st.session_state.pitches), template.row if template.row > 0 else 1)
            st.session_state.fasteners.append(
                replace(
                    template,
                    row=max(1, default_row) if n_rows > 0 else 1,
                )
            )
    with fcols[1]:
        st.write("Configure any number of fasteners and assign them to available nodes.")

    remove_fasteners: List[int] = []
    methods = ["Boeing69", "Huth_metal", "Huth_graphite", "Grumman", "Manual"]
    for idx, fastener in enumerate(st.session_state.fasteners):
        with st.expander(f"Fastener {idx + 1} — node {fastener.row}", expanded=(len(st.session_state.fasteners) <= 6)):
            c0, c1, c2, c3, c4 = st.columns([1, 1, 1, 1, 0.5])
            clamped_row = min(max(int(fastener.row), 1), max(n_rows, 1))
            fastener.row = int(
                c0.number_input(
                    "Node index", 1, max(n_rows, 1), clamped_row, key=f"fr_row_{idx}", step=1
                )
            )
            fastener.D = c1.number_input("Diameter d [in]", 0.01, 2.0, fastener.D, key=f"fr_d_{idx}", step=0.001, format="%.3f")
            fastener.Eb = c2.number_input("Bolt E [psi]", 1e5, 5e8, fastener.Eb, key=f"fr_Eb_{idx}", step=1e5, format="%.0f")
            fastener.nu_b = c3.number_input("Bolt ν", 0.0, 0.49, fastener.nu_b, key=f"fr_nu_{idx}", step=0.01, format="%.2f")
            fastener.method = c4.selectbox("Method", methods, index=methods.index(fastener.method), key=f"fr_m_{idx}")
            if fastener.method == "Manual":
                fastener.k_manual = st.number_input(
                    "Manual k [lb/in]", 1.0, 1e12, fastener.k_manual or 1.0e6, key=f"fr_km_{idx}", step=1e5, format="%.0f"
                )
            if st.button("✖ Remove", key=f"fr_rm_{idx}"):
                remove_fasteners.append(idx)
    if remove_fasteners:
        for idx in sorted(remove_fasteners, reverse=True):
            if 0 <= idx < len(st.session_state.fasteners):
                st.session_state.fasteners.pop(idx)

    st.divider()
    st.subheader("Supports (Dirichlet u=0)")
    if st.button("➕ Add support"):
        st.session_state.supports.append((0, 0, 0.0))
    remove_ids: List[int] = []
    for idx, (plate_idx, local_node, value) in enumerate(st.session_state.supports):
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            max_plate_idx = max(0, len(st.session_state.plates) - 1)
            clamped_plate_idx = min(max(int(plate_idx), 0), max_plate_idx)
            plate_idx = c1.number_input(
                f"Support {idx} — Plate index (0..)", 0, max_plate_idx, clamped_plate_idx, key=f"sp_pi_{idx}"
            )
            selected_plate = st.session_state.plates[int(plate_idx)]
            segments = selected_plate.last_row - selected_plate.first_row + 1
            clamped_local = min(max(int(local_node), 0), segments)
            local_node = c2.number_input("Local node (0..nSeg)", 0, segments, clamped_local, key=f"sp_ln_{idx}")
            value = c3.number_input("u [in]", -1.0, 1.0, float(value), key=f"sp_val_{idx}", step=0.001, format="%.3f")
            st.session_state.supports[idx] = (int(plate_idx), int(local_node), float(value))
            if c4.button("✖", key=f"sp_rm_{idx}"):
                remove_ids.append(idx)
    if remove_ids:
        for idx in sorted(remove_ids, reverse=True):
            st.session_state.supports.pop(idx)

    st.divider()
    if st.button("Load ▶ JOLT Figure 76"):
        (
            st.session_state.pitches,
            st.session_state.plates,
            st.session_state.fasteners,
            st.session_state.supports,
        ) = load_example_figure76()


# ---------- 2.4 Solution ----------

st.title("JOLT 1D Joint — Bars + Springs")

pitches = st.session_state.pitches
plates = st.session_state.plates
fasteners = st.session_state.fasteners
supports = st.session_state.supports

if st.button("Solve", type="primary"):
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports, point_forces=None)

    if plt is not None:
        spacing = 120.0
        y_levels = {idx: -idx * spacing for idx in range(len(plates))}
        fig, ax = plt.subplots(figsize=(12, 4 + max(len(plates) - 2, 0) * 0.5))
        global_nodes: List[float] = [0.0]
        for pitch in pitches:
            global_nodes.append(global_nodes[-1] + pitch)
        for xi in global_nodes:
            ax.axvline(x=xi, ymin=0.05, ymax=0.95, ls=":", lw=0.6, color="0.8")

        node_coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
        max_pitch = max(pitches) if pitches else 1.0
        arrow_length = max(0.3, 0.35 * max_pitch)

        for plate_idx, plate in enumerate(plates):
            segments = plate.last_row - plate.first_row + 1
            start_index = max(plate.first_row - 1, 0)
            coords = [global_nodes[start_index]]
            for seg in range(segments):
                coords.append(coords[-1] + pitches[start_index + seg])
            y = y_levels[plate_idx]
            ys = [y] * (segments + 1)
            ax.plot(coords, ys, marker="o", lw=2, label=plate.name)
            for local_node, x_node in enumerate(coords):
                node_coords[(plate_idx, local_node)] = (x_node, y)
                ax.scatter(x_node, y, s=36, c="white", edgecolors="black", zorder=5)
                ax.text(x_node, y + 14, f"n{local_node}", ha="center", va="bottom", fontsize=8)

            if abs(plate.Fx_left) > 0.0:
                direction = 1 if plate.Fx_left >= 0 else -1
                x_start = coords[0]
                x_end = x_start + direction * arrow_length
                ax.annotate(
                    "",
                    xy=(x_end, y),
                    xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="<|-", color="tab:red", lw=2),
                )
                ax.scatter(x_start, y, marker="s", s=60, c="tab:red", zorder=6)
                ax.text(
                    x_end + direction * 0.05 * max_pitch,
                    y + 26,
                    f"{plate.Fx_left:+.0f} lb",
                    ha="left" if direction >= 0 else "right",
                    va="bottom",
                    fontsize=9,
                    color="tab:red",
                    fontweight="bold",
                )
            if abs(plate.Fx_right) > 0.0:
                direction = 1 if plate.Fx_right >= 0 else -1
                x_start = coords[-1]
                x_end = x_start + direction * arrow_length
                ax.annotate(
                    "",
                    xy=(x_end, y),
                    xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="<|-", color="tab:red", lw=2),
                )
                ax.scatter(x_start, y, marker="s", s=60, c="tab:red", zorder=6)
                ax.text(
                    x_end + direction * 0.05 * max_pitch,
                    y + 26,
                    f"{plate.Fx_right:+.0f} lb",
                    ha="left" if direction >= 0 else "right",
                    va="bottom",
                    fontsize=9,
                    color="tab:red",
                    fontweight="bold",
                )

        for s_idx, (plate_idx, local_node, value) in enumerate(supports):
            coord = node_coords.get((plate_idx, local_node))
            if coord is None:
                continue
            x_node, y_node = coord
            ax.plot([x_node, x_node], [y_node, y_node - 18], color="tab:green", lw=2, zorder=4)
            ax.scatter(x_node, y_node - 18, marker="v", s=90, c="tab:green", zorder=5)
            ax.text(
                x_node,
                y_node - 32,
                f"S{s_idx}: {plates[plate_idx].name} n{local_node}\n u={value:+.3f} in",
                ha="center",
                va="top",
                fontsize=8,
                color="tab:green",
            )

        for fast_idx, fastener in enumerate(fasteners):
            row_index = int(fastener.row)
            if not (1 <= row_index <= len(pitches)):
                continue
            attachments: List[Tuple[float, float]] = []
            for plate_idx, plate in enumerate(plates):
                if plate.first_row <= row_index <= plate.last_row:
                    local = row_index - plate.first_row
                    if 0 <= local <= plate.last_row - plate.first_row:
                        coord = node_coords.get((plate_idx, local))
                        if coord is not None:
                            attachments.append(coord)
            if len(attachments) < 2:
                continue
            attachments.sort(key=lambda item: item[1], reverse=True)
            xs_attach = [item[0] for item in attachments]
            ys_attach = [item[1] for item in attachments]
            ax.plot(xs_attach, ys_attach, ls="--", color="tab:purple", lw=1.5, zorder=4)
            ax.scatter(xs_attach, ys_attach, c="tab:purple", s=40, zorder=5)
            ax.text(
                sum(xs_attach) / len(xs_attach),
                max(ys_attach) + 18,
                f"F{fast_idx + 1} @ n{row_index}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:purple",
            )

        ax.set_xlabel("x [in]")
        if plates:
            ax.set_yticks([y_levels[i] for i in range(len(plates))])
            ax.set_yticklabels([plate.name for plate in plates])
        ax.set_ylim(min(y_levels.values(), default=0.0) - 80.0, 60.0)
        handles, labels = ax.get_legend_handles_labels()
        if Line2D is not None:
            extra_handles = [
                Line2D([0], [0], ls="--", color="tab:purple", lw=1.5, label="Fastener"),
                Line2D([0], [0], marker="v", color="tab:green", linestyle="", markersize=10, label="Support"),
                Line2D([0], [0], marker="s", color="tab:red", linestyle="", markersize=8, label="Load"),
            ]
            handles += extra_handles
            labels += [h.get_label() for h in extra_handles]
        ax.legend(handles, labels, loc="upper right")
        ax.set_title("Scheme overview (nodes n#, supports S#, fasteners F#)")
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
    else:  # pragma: no cover - executed only when matplotlib is unavailable
        st.info("Matplotlib is not installed. Install it to see the schematic plot.")

    st.subheader("Fasteners")
    fastener_dicts = solution.fasteners_as_dicts()
    if pd is not None:
        df_fast = pd.DataFrame(fastener_dicts)
        st.dataframe(df_fast.style.format({"CF [in/lb]": "{:.3e}", "k [lb/in]": "{:.3e}", "F [lb]": "{:.2f}"}))
    else:  # pragma: no cover
        st.table(fastener_dicts)

    st.subheader("Nodes")
    node_dicts = _node_table(solution)
    if pd is not None:
        df_nodes = pd.DataFrame(node_dicts)
        st.dataframe(
            df_nodes.style.format(
                {
                    "X [in]": "{:.3f}",
                    "u [in]": "{:.6e}",
                    "Net Bypass [lb]": "{:.2f}",
                    "t [in]": "{:.3f}",
                    "Bypass Area [in^2]": "{:.3f}",
                }
            )
        )
    else:  # pragma: no cover
        st.table(node_dicts)

    st.subheader("Bars (plate segments)")
    bar_dicts = _bar_table(solution)
    if pd is not None:
        df_bars = pd.DataFrame(bar_dicts)
        st.dataframe(df_bars.style.format({"Force [lb]": "{:.2f}", "k_bar [lb/in]": "{:.3e}", "E [psi]": "{:.3e}"}))
    else:  # pragma: no cover
        st.table(bar_dicts)

    st.subheader("Bearing / Bypass by row & plate")
    bearing_dicts = _bearing_table(solution)
    if pd is not None:
        df_bb = pd.DataFrame(bearing_dicts)
        st.dataframe(df_bb.style.format({"Bearing [lb]": "{:.2f}", "Bypass [lb]": "{:.2f}"}))
    else:  # pragma: no cover
        st.table(bearing_dicts)

    if pd is not None:
        st.download_button("Export fasteners CSV", data=df_fast.to_csv(index=False).encode("utf-8"), file_name="fasteners.csv", mime="text/csv")
        st.download_button("Export nodes CSV", data=df_nodes.to_csv(index=False).encode("utf-8"), file_name="nodes.csv", mime="text/csv")
        st.download_button("Export bars CSV", data=df_bars.to_csv(index=False).encode("utf-8"), file_name="bars.csv", mime="text/csv")
        st.download_button(
            "Export bearing_bypass CSV",
            data=df_bb.to_csv(index=False).encode("utf-8"),
            file_name="bearing_bypass.csv",
            mime="text/csv",
        )
else:
    st.info("Соберите схему в левой панели и нажмите **Solve**. Для воспроизведения скрина используйте **Load ▶ JOLT Figure 76**.")
