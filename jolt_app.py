"""Streamlit application for the JOLT 1D joint model."""
from __future__ import annotations

from typing import List

try:  # Optional imports used for the UI only
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional for tests
    plt = None  # type: ignore

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
    st.subheader("Fasteners by row")
    for idx, fastener in enumerate(st.session_state.fasteners):
        with st.expander(f"Row {fastener.row}", expanded=(len(st.session_state.fasteners) <= 6)):
            c1, c2, c3, c4 = st.columns(4)
            fastener.D = c1.number_input("Diameter d [in]", 0.01, 2.0, fastener.D, key=f"fr_d_{idx}", step=0.001, format="%.3f")
            fastener.Eb = c2.number_input("Bolt E [psi]", 1e5, 5e8, fastener.Eb, key=f"fr_Eb_{idx}", step=1e5, format="%.0f")
            fastener.nu_b = c3.number_input("Bolt ν", 0.0, 0.49, fastener.nu_b, key=f"fr_nu_{idx}", step=0.01, format="%.2f")
            methods = ["Boeing69", "Huth_metal", "Huth_graphite", "Grumman", "Manual"]
            fastener.method = c4.selectbox("Method", methods, index=methods.index(fastener.method), key=f"fr_m_{idx}")
            if fastener.method == "Manual":
                fastener.k_manual = st.number_input(
                    "Manual k [lb/in]", 1.0, 1e12, fastener.k_manual or 1.0e6, key=f"fr_km_{idx}", step=1e5, format="%.0f"
                )
    if len(st.session_state.fasteners) != n_rows:
        if len(st.session_state.fasteners) < n_rows:
            template = st.session_state.fasteners[-1]
            for row in range(len(st.session_state.fasteners) + 1, n_rows + 1):
                st.session_state.fasteners.append(
                    FastenerRow(row=row, D=template.D, Eb=template.Eb, nu_b=template.nu_b, method=template.method)
                )
        else:
            st.session_state.fasteners = st.session_state.fasteners[:n_rows]

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
        y_levels = {idx: -idx * 100.0 for idx in range(len(plates))}
        fig, ax = plt.subplots(figsize=(10, 4))
        x = 0.0
        xs = [0.0]
        for pitch in pitches:
            x += pitch
            xs.append(x)
        for xi in xs:
            ax.axvline(x=xi, ymin=0.05, ymax=0.95, ls=":", lw=0.5)
        for plate_idx, plate in enumerate(plates):
            segments = plate.last_row - plate.first_row + 1
            x0 = sum(pitches[: plate.first_row - 1])
            coords = [x0]
            for seg in range(segments):
                coords.append(coords[-1] + pitches[plate.first_row - 1 + seg])
            ys = [y_levels[plate_idx]] * (segments + 1)
            ax.plot(coords, ys, marker="o", lw=2, label=plate.name)
            if abs(plate.Fx_left) > 0.0:
                ax.arrow(coords[0] - 0.2, ys[0], 0.15, 0.0, head_width=10, head_length=0.15, length_includes_head=True)
                ax.text(coords[0] - 0.25, ys[0] + 12, f"{plate.Fx_left:.0f} lb", ha="right", va="bottom")
            if abs(plate.Fx_right) > 0.0:
                ax.arrow(coords[-1] + 0.2, ys[-1], -0.15, 0.0, head_width=10, head_length=0.15, length_includes_head=True)
                ax.text(coords[-1] + 0.25, ys[-1] + 12, f"{plate.Fx_right:.0f} lb", ha="left", va="bottom")
            for supp_plate, supp_node, _ in supports:
                if supp_plate == plate_idx:
                    ax.plot(coords[supp_node], ys[supp_node] - 12, marker="^", ms=10)
        for fastener in fasteners:
            row = fastener.row
            present = [idx for idx, plate in enumerate(plates) if plate.first_row <= row <= plate.last_row]
            present.sort()
            x_row = sum(pitches[: row - 1])
            for upper, lower in zip(present[:-1], present[1:]):
                ax.plot([x_row, x_row], [y_levels[upper], y_levels[lower]], ls="--", lw=1)
        ax.set_xlabel("x [in]")
        ax.set_yticks([y_levels[i] for i in range(len(plates))])
        ax.set_yticklabels([plate.name for plate in plates])
        ax.legend(loc="upper right")
        ax.set_title("User-defined scheme (nodes •, supports ▲, fasteners --)")
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
