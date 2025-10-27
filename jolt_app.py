# jolt_app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Literal
import numpy as np
import pandas as pd

# --- UI (Streamlit) ---
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# 1) ФИЗИЧЕСКИЙ ДВИЖОК: 1‑D Joint (бары + пружины)
#    Модель соответствует общепринятой 1D‑схеме:
#    - Стержни: k_bar = E*A/L
#    - Пружины крепежа: k_fast = 1 / CF   (CF — податливость)
#    Сборка глобальной [K], ГУ типа u=const, решение [K]{u}={P}
# ============================================================

# ---------- 1.1 Податливость крепежа (формулы) ----------

def boeing69_CF(ti: float, Ei: float,
                tj: float, Ej: float,
                Eb: float, nu_b: float, D: float) -> float:
    """
    Boeing (1969, D6-29942): Fastener Spring Constant expressed as COMPLIANCE [in/lb]
      C_F = 4(ti+tj)/(9 Gb Ab) + (ti^3 + 5 ti^2 tj + 5 ti tj^2 + tj^3)/(40 Eb I_b)
            + (1/ti)*(1/Eb + 1/Ei) + (1/tj)*(1/Eb + 1/Ej)
    где Ab = π D^2 / 4,   I_b = π D^4 / 64,   Gb = Eb / (2(1+ν_b))
    Возвращает CF [in/lb]. Жесткость k = 1/CF [lb/in].
    """
    Ab = math.pi * D**2 / 4.0
    Ib = math.pi * D**4 / 64.0
    Gb = Eb / (2.0 * (1.0 + nu_b))
    term_shear   = 4.0 * (ti + tj) / (9.0 * Gb * Ab)
    term_bending = (ti**3 + 5.0*ti**2*tj + 5.0*ti*tj**2 + tj**3) / (40.0 * Eb * Ib)
    term_bearing = (1.0/ti) * (1.0/Eb + 1.0/Ei) + (1.0/tj) * (1.0/Eb + 1.0/Ej)
    return term_shear + term_bending + term_bearing  # [in/lb]

def huth_CF(ti: float, Ei: float,
            tj: float, Ej: float,
            Ef: float, d: float,
            shear: Literal["single","double"]="single",
            joint_type: Literal["bolted_metal","riveted_metal","bolted_graphite"]="bolted_metal") -> float:
    """
    Huth (ASTM STP 927). Eq. (2.14) у Soderberg.
      f = ((t1+t2)/(2d))^a * (b/n) * ( 1/(t1E1) + 1/(n t2 E2) + 1/(2 t1 Ef) + 1/(2 n t2 Ef) )
    где n = 1 (single) или 2 (double); параметры a,b — по типу соединения.
    Возвращает f [in/lb].
    """
    if joint_type == "bolted_metal":
        a, b = (2.0/3.0, 3.0)
    elif joint_type == "riveted_metal":
        a, b = (2.0/5.0, 2.2)
    else:  # "bolted_graphite"
        a, b = (2.0/3.0, 4.2)
    n = 1.0 if shear == "single" else 2.0
    geom = ((ti + tj) / (2.0 * d))**a
    core = (1.0/(ti*Ei) + 1.0/(n*tj*Ej) + 1.0/(2.0*ti*Ef) + 1.0/(2.0*n*tj*Ef))
    f = geom * (b/n) * core
    return f  # [in/lb]

def grumman_CF(ti: float, Ei: float, tj: float, Ej: float, Ef: float, d: float) -> float:
    """
    Grumman (эмпирическая) — одноножневая геометрия.
      f = (t1+t2)^2 / (Ef * d) + 3.72 * ( 1/(E1 t1) + 1/(E2 t2) )
    Возвращает f [in/lb].
    """
    return (ti + tj)**2 / (Ef * d) + 3.72 * (1.0/(Ei*ti) + 1.0/(Ej*tj))

# ---------- 1.2 Данные модели ----------

@dataclass
class Plate:
    name: str
    E: float              # [psi]
    t: float              # [in]
    first_row: int        # 1..N
    last_row: int         # 1..N
    # Площадь полосы по сегментам (между рядами) — len = last-first+1
    A_strip: List[float]  # [in^2]
    Fx_left: float = 0.0  # [lb] внешние силы в концах слоя
    Fx_right: float = 0.0

@dataclass
class FastenerRow:
    row: int
    # описание болта
    D: float           # [in]
    Eb: float          # [psi]
    nu_b: float        # [-]
    method: str = "Boeing69"  # "Boeing69"|"Huth_metal"|"Huth_graphite"|"Grumman"|"Manual"
    k_manual: Optional[float] = None  # [lb/in] если method=="Manual"

@dataclass
class Joint1D:
    pitches: List[float]            # длины между рядами, len = N_rows
    plates: List[Plate]
    fasteners: List[FastenerRow]    # один объект на каждый ряд
    # служебные
    _dof: Dict[Tuple[int,int], int] = field(init=False, default_factory=dict)
    _x:   Dict[Tuple[int,int], float] = field(init=False, default_factory=dict)

    # --- разметка степеней свободы ---
    def _build_dofs(self) -> int:
        self._dof.clear(); self._x.clear()
        ndof = 0
        for p_idx,p in enumerate(self.plates):
            x0 = sum(self.pitches[:p.first_row-1])
            xs = [x0]
            for seg in range(p.first_row-1, p.last_row):
                xs.append(xs[-1] + self.pitches[seg])
            for ln,x in enumerate(xs):
                self._dof[(p_idx, ln)] = ndof
                self._x[(p_idx, ln)]   = x
                ndof += 1
        return ndof

    def _CF_for_pair(self, fr: FastenerRow, pi: Plate, pj: Plate) -> Tuple[float,float]:
        """Возвратить (CF, k) для пары соседних слоёв (pi,pj) на ряду fr.row."""
        if fr.method == "Manual" and fr.k_manual is not None:
            k = float(fr.k_manual); CF = 1.0/k if k>0.0 else 1e12
            return CF, k
        if fr.method == "Boeing69":
            CF = boeing69_CF(pi.t, pi.E, pj.t, pj.E, fr.Eb, fr.nu_b, fr.D)
        elif fr.method == "Huth_metal":
            CF = huth_CF(pi.t, pi.E, pj.t, pj.E, fr.Eb, fr.D, "single", "bolted_metal")
        elif fr.method == "Huth_graphite":
            CF = huth_CF(pi.t, pi.E, pj.t, pj.E, fr.Eb, fr.D, "single", "bolted_graphite")
        elif fr.method == "Grumman":
            CF = grumman_CF(pi.t, pi.E, pj.t, pj.E, fr.Eb, fr.D)
        else:  # fallback
            CF = boeing69_CF(pi.t, pi.E, pj.t, pj.E, fr.Eb, fr.nu_b, fr.D)
        k = 1.0/CF if CF>0.0 else 1e12
        return CF, k

    # --- решение ---
    def solve(self, supports: List[Tuple[int,int,float]],
              point_forces: List[Tuple[int,int,float]] = None) -> Dict:
        """
        supports: список закреплений (plate_index, local_node, u=значение), обычно u=0.
        point_forces: узловые силы (plate_index, local_node, Fx)
        """
        N_rows = len(self.pitches)
        ndof = self._build_dofs()
        K = np.zeros((ndof, ndof))
        P = np.zeros(ndof)

        # 1) Стержни (EA/L) по каждому сегменту слоя
        for p_idx,p in enumerate(self.plates):
            nSeg = p.last_row - p.first_row + 1
            assert len(p.A_strip)==nSeg, f"{p.name}: A_strip len must be {nSeg}"
            for s in range(nSeg):
                Lseg = self.pitches[p.first_row-1 + s]
                Aseg = p.A_strip[s]
                kbar = p.E * Aseg / Lseg
                iL, iR = s, s+1
                dofL = self._dof[(p_idx, iL)]
                dofR = self._dof[(p_idx, iR)]
                # матрица 2x2
                K[dofL,dofL] += kbar
                K[dofL,dofR] -= kbar
                K[dofR,dofL] -= kbar
                K[dofR,dofR] += kbar
            # внешние силы на концах слоя
            dofL = self._dof[(p_idx, 0)]
            dofR = self._dof[(p_idx, nSeg)]
            P[dofL] += p.Fx_left
            P[dofR] += p.Fx_right

        # 2) Пружины‑крепежи (по каждому ряду и паре соседних слоёв)
        springs = []  # (i_dof, j_dof, k, row, iface, CF)
        for fr in self.fasteners:
            r = fr.row
            present = [pi for pi,p in enumerate(self.plates) if p.first_row <= r <= p.last_row]
            present.sort()
            # пары соседних слоёв: (0,1), (1,2), ...
            for a,b in zip(present[:-1], present[1:]):
                pi = self.plates[a]; pj = self.plates[b]
                ln_i = r - pi.first_row
                ln_j = r - pj.first_row
                dof_i = self._dof[(a, ln_i)]
                dof_j = self._dof[(b, ln_j)]
                CF, k = self._CF_for_pair(fr, pi, pj)
                # вклад пружины
                K[dof_i,dof_i] += k
                K[dof_i,dof_j] -= k
                K[dof_j,dof_i] -= k
                K[dof_j,dof_j] += k
                springs.append((dof_i, dof_j, k, r, f"{pi.name}-{pj.name}", CF))

        # 3) Узловые силы (если заданы)
        if point_forces:
            for (pi, ln, Fx) in point_forces:
                dof = self._dof[(pi, ln)]
                P[dof] += Fx

        # 4) ГУ (Dirichlet u=const)
        fixed_dofs = []
        u_fixed = []
        for (pi, ln, val) in supports:
            fixed_dofs.append(self._dof[(pi, ln)])
            u_fixed.append(val)
        fixed_dofs = np.array(fixed_dofs, dtype=int)
        u_fixed    = np.array(u_fixed, dtype=float)

        all_idx = np.arange(ndof, dtype=int)
        free_mask = np.ones(ndof, dtype=bool)
        free_mask[fixed_dofs] = False
        free = all_idx[free_mask]

        # 5) Решение K_ff u_f = P_f - K_fp u_p
        K_ff = K[np.ix_(free, free)]
        K_fp = K[np.ix_(free, fixed_dofs)]
        P_f  = P[free]
        rhs  = P_f - K_fp @ u_fixed
        u_f  = np.linalg.solve(K_ff, rhs)

        u = np.zeros(ndof)
        u[free] = u_f
        u[fixed_dofs] = u_fixed

        # 6) Усилия в пружинах
        fast_tbl = []
        for (i,j,k,row,iface,CF) in springs:
            F = k * (u[i] - u[j])  # знак: из "верхней" в "нижнюю" пластину пары
            fast_tbl.append((row, iface, CF, k, F, i, j))

        # 7) Bearing/Bypass по ряду/слою (по двум смежным сегментам)
        bb_tbl = []
        for r in range(1, len(self.pitches)+1):
            for p_idx,p in enumerate(self.plates):
                if not (p.first_row <= r <= p.last_row):
                    continue
                nSeg = p.last_row - p.first_row + 1
                # поток по левому сегменту (если есть)
                F_left = 0.0
                sL = r-1 - p.first_row
                if sL >= 0:
                    Lseg = self.pitches[p.first_row-1 + sL]
                    Aseg = p.A_strip[sL]
                    dofL = self._dof[(p_idx, sL)]
                    dofR = self._dof[(p_idx, sL+1)]
                    kbar = p.E * Aseg / Lseg
                    F_left = kbar * (u[dofR] - u[dofL])
                # поток по правому сегменту (если есть)
                F_right = 0.0
                sR = r - p.first_row
                if sR < nSeg:
                    Lseg = self.pitches[p.first_row-1 + sR]
                    Aseg = p.A_strip[sR]
                    dofL = self._dof[(p_idx, sR)]
                    dofR = self._dof[(p_idx, sR+1)]
                    kbar = p.E * Aseg / Lseg
                    F_right = kbar * (u[dofR] - u[dofL])
                F_bearing = F_right - F_left
                F_bypass  = F_right if (F_left > F_right) else F_left
                bb_tbl.append((r, self.plates[p_idx].name, F_bearing, F_bypass))

        # 8) Узлы/геометрия — для таблиц
        node_tbl = []
        for p_idx,p in enumerate(self.plates):
            nSeg = p.last_row - p.first_row + 1
            for ln in range(nSeg+1):
                dof = self._dof[(p_idx, ln)]
                x   = self._x[(p_idx, ln)]
                # "net bypass" как средняя из соседних потоков (для наглядности)
                nb = 0.0
                if ln>0:
                    Lseg = self.pitches[p.first_row-1 + (ln-1)]
                    Aseg = p.A_strip[ln-1]
                    kbar = p.E * Aseg / Lseg
                    nb += kbar * (u[self._dof[(p_idx, ln)]] - u[self._dof[(p_idx, ln-1)]])
                if ln<nSeg:
                    Lseg = self.pitches[p.first_row-1 + ln]
                    Aseg = p.A_strip[ln]
                    kbar = p.E * Aseg / Lseg
                    nb += kbar * (u[self._dof[(p_idx, ln+1)]] - u[self._dof[(p_idx, ln)]])
                nb *= 0.5
                node_tbl.append((p_idx, p.name, ln, x, u[dof], nb, p.t,
                                 (p.A_strip[ln] if ln < nSeg else p.A_strip[-1])))

        # 9) Сегменты (стержни) — для таблиц
        bar_tbl = []
        for p_idx,p in enumerate(self.plates):
            nSeg = p.last_row - p.first_row + 1
            for s in range(nSeg):
                Lseg = self.pitches[p.first_row-1 + s]
                Aseg = p.A_strip[s]
                kbar = p.E * Aseg / Lseg
                dofL = self._dof[(p_idx, s)]
                dofR = self._dof[(p_idx, s+1)]
                F    = kbar * (u[dofR] - u[dofL])
                bar_tbl.append((p_idx, p.name, s, F, kbar, p.E))

        return dict(
            u=u, K=K, P=P,
            fasteners=sorted(fast_tbl),
            bearing_bypass=bb_tbl,
            node_table=node_tbl,
            bar_table=bar_tbl,
            dof_map=self._dof,
        )

# ============================================================
# 2) UI‑СЛОЙ (Streamlit)
# ============================================================

st.set_page_config(page_title="JOLT 1D Joint", layout="wide")

# ---------- 2.1 Сервис: шаблон примера Figure 76 ----------
def load_example_figure76():
    # 7 рядов, одинаковый шаг
    pitches = [1.128]*7
    E_sheet = 1.05e7    # psi (алюминий в ваших исходниках)
    E_bolt  = 1.04e7    # psi
    nu_bolt = 0.30
    D = 0.188           # in
    plates = [
        Plate(name="Tripler", E=E_sheet, t=0.083, first_row=1, last_row=3,
              A_strip=[0.071,0.071,0.071], Fx_left=0.0, Fx_right=0.0),
        Plate(name="Doubler", E=E_sheet, t=0.040, first_row=1, last_row=7,
              A_strip=[0.045]*7, Fx_left=0.0, Fx_right=0.0),
        Plate(name="Skin",    E=E_sheet, t=0.040, first_row=4, last_row=7,
              A_strip=[0.045]*4, Fx_left=+1000.0, Fx_right=0.0),  # 1000 lb влево на Skin слева
    ]
    fasteners = [FastenerRow(row=r, D=D, Eb=E_bolt, nu_b=nu_bolt, method="Boeing69") for r in range(1,8)]
    # Опоры: правые концы Tripler и Doubler
    supports = [
        (0, 3, 0.0),   # Tripler: локальный узел 3 (правый конец слоя 1..3)
        (1, 7, 0.0),   # Doubler: локальный узел 7
    ]
    return pitches, plates, fasteners, supports

# ---------- 2.2 Состояние ----------
if "pitches" not in st.session_state:
    st.session_state.pitches, st.session_state.plates, st.session_state.fasteners, st.session_state.supports = load_example_figure76()

# ---------- 2.3 Вводные панели ----------
with st.sidebar:
    st.header("Geometry")
    # ряды и шаги
    n_rows = st.number_input("Number of rows", 1, 50, len(st.session_state.pitches))
    # редактирование шагов
    if n_rows != len(st.session_state.pitches):
        # растянуть/урезать, заполняя последним значением
        if n_rows > len(st.session_state.pitches):
            st.session_state.pitches = st.session_state.pitches + [st.session_state.pitches[-1]]*(n_rows - len(st.session_state.pitches))
        else:
            st.session_state.pitches = st.session_state.pitches[:n_rows]
    cols = st.columns(2)
    with cols[0]:
        same_pitch = st.checkbox("All pitches equal", value=True)
    if same_pitch:
        pval = st.number_input("Pitch value [in]", 0.01, 100.0, st.session_state.pitches[0], step=0.001, format="%.3f")
        st.session_state.pitches = [float(pval)]*n_rows
    else:
        st.write("Pitches [in]")
        # компактный редактор шага
        pitch_vals = []
        for i in range(n_rows):
            pitch_vals.append(st.number_input(f"p[{i+1}]", 0.001, 100.0, st.session_state.pitches[i], key=f"pitch_{i}", step=0.001, format="%.3f"))
        st.session_state.pitches = pitch_vals

    st.divider()
    st.subheader("Plates (Layers)")
    # редактор слоёв
    for idx, p in enumerate(st.session_state.plates):
        with st.expander(f"Layer {idx}: {p.name}", expanded=False):
            c1,c2,c3 = st.columns(3)
            p.name = c1.text_input("Name", p.name, key=f"pl_name_{idx}")
            p.E    = c2.number_input("E [psi]", 1e5, 5e8, p.E, key=f"pl_E_{idx}", step=1e5, format="%.0f")
            p.t    = c3.number_input("t [in]", 0.001, 2.0, p.t, key=f"pl_t_{idx}", step=0.001, format="%.3f")
            d1,d2,d3 = st.columns(3)
            p.first_row = int(d1.number_input("First row", 1, n_rows, p.first_row, key=f"pl_fr_{idx}"))
            p.last_row  = int(d2.number_input("Last row",  1, n_rows, p.last_row, key=f"pl_lr_{idx}"))
            st.write(f"Slices (segments) = {p.last_row - p.first_row + 1}")
            # площади полос по сегментам
            nSeg = p.last_row - p.first_row + 1
            if len(p.A_strip) != nSeg:
                p.A_strip = [p.A_strip[0] if p.A_strip else 0.05]*nSeg
            if st.checkbox("Same bypass area for all segments", value=True, key=f"sameA_{idx}"):
                aval = st.number_input("Bypass area per segment [in²]", 1e-5, 10.0, p.A_strip[0], key=f"pl_A_all_{idx}", step=0.001, format="%.3f")
                p.A_strip = [float(aval)]*nSeg
            else:
                for s in range(nSeg):
                    p.A_strip[s] = st.number_input(f"A[{s+1}] [in²]", 1e-5, 10.0, p.A_strip[s], key=f"pl_A_{idx}_{s}", step=0.001, format="%.3f")
            e1,e2 = st.columns(2)
            p.Fx_left  = e1.number_input("End load LEFT [+→] [lb]", -1e6, 1e6, p.Fx_left, key=f"pl_Fl_{idx}", step=1.0, format="%.1f")
            p.Fx_right = e2.number_input("End load RIGHT [+→] [lb]", -1e6, 1e6, p.Fx_right, key=f"pl_Fr_{idx}", step=1.0, format="%.1f")
    cadd, cex = st.columns([1,1])
    if cadd.button("➕ Add layer"):
        st.session_state.plates.append(
            Plate(name=f"Layer{len(st.session_state.plates)}", E=1.0e7, t=0.05, first_row=1, last_row=n_rows,
                  A_strip=[0.05]*(n_rows), Fx_left=0.0, Fx_right=0.0)
        )
    if cex.button("🗑 Remove last layer") and len(st.session_state.plates)>1:
        st.session_state.plates.pop()

    st.divider()
    st.subheader("Fasteners by row")
    for i, fr in enumerate(st.session_state.fasteners):
        with st.expander(f"Row {fr.row}", expanded=(len(st.session_state.fasteners)<=6)):
            c1,c2,c3,c4 = st.columns(4)
            fr.D   = c1.number_input("Diameter d [in]", 0.01, 2.0, fr.D, key=f"fr_d_{i}", step=0.001, format="%.3f")
            fr.Eb  = c2.number_input("Bolt E [psi]", 1e5, 5e8, fr.Eb, key=f"fr_Eb_{i}", step=1e5, format="%.0f")
            fr.nu_b= c3.number_input("Bolt ν", 0.0, 0.49, fr.nu_b, key=f"fr_nu_{i}", step=0.01, format="%.2f")
            fr.method = c4.selectbox("Method", ["Boeing69","Huth_metal","Huth_graphite","Grumman","Manual"],
                                     index=["Boeing69","Huth_metal","Huth_graphite","Grumman","Manual"].index(fr.method),
                                     key=f"fr_m_{i}")
            if fr.method=="Manual":
                fr.k_manual = st.number_input("Manual k [lb/in]", 1.0, 1e12, fr.k_manual or 1.0e6, key=f"fr_km_{i}", step=1e5, format="%.0f")
    # синхронизировать число рядов fasteners
    if len(st.session_state.fasteners) != n_rows:
        if len(st.session_state.fasteners) < n_rows:
            last = st.session_state.fasteners[-1]
            for r in range(len(st.session_state.fasteners)+1, n_rows+1):
                st.session_state.fasteners.append(FastenerRow(row=r, D=last.D, Eb=last.Eb, nu_b=last.nu_b, method=last.method))
        else:
            st.session_state.fasteners = st.session_state.fasteners[:n_rows]

    st.divider()
    st.subheader("Supports (Dirichlet u=0)")
    if st.button("➕ Add support"):
        st.session_state.supports.append((0,0,0.0))
    rm_ids = []
    for i,(pi,ln,val) in enumerate(st.session_state.supports):
        with st.container():
            c1,c2,c3,c4 = st.columns([2,2,2,1])
            pi = c1.number_input(f"Support {i} — Plate index (0..)", 0, max(0,len(st.session_state.plates)-1), int(pi), key=f"sp_pi_{i}")
            # вычислить число локальных узлов выбранного слоя
            nSeg = st.session_state.plates[int(pi)].last_row - st.session_state.plates[int(pi)].first_row + 1
            ln = c2.number_input("Local node (0..nSeg)", 0, nSeg, int(ln), key=f"sp_ln_{i}")
            val = c3.number_input("u [in]", -1.0, 1.0, float(val), key=f"sp_val_{i}", step=0.0)
            st.session_state.supports[i] = (int(pi), int(ln), float(val))
            if c4.button("✖", key=f"sp_rm_{i}"):
                rm_ids.append(i)
    if rm_ids:
        for i in sorted(rm_ids, reverse=True):
            st.session_state.supports.pop(i)

    st.divider()
    if st.button("Load ▶ JOLT Figure 76"):
        st.session_state.pitches, st.session_state.plates, st.session_state.fasteners, st.session_state.supports = load_example_figure76()

# ---------- 2.4 Решение ----------
st.title("JOLT 1D Joint — Bars + Springs")
pitches = st.session_state.pitches
plates  = st.session_state.plates
fasteners = st.session_state.fasteners
supports  = st.session_state.supports

if st.button("Solve", type="primary"):
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    out = model.solve(supports=supports, point_forces=None)

    # --------- Визуализация схемы ----------
    y_levels = {}
    y = 0.0
    for i,p in enumerate(plates):
        y_levels[i] = -i*100.0  # просто вертикальное разведение
    fig, ax = plt.subplots(figsize=(10,4))
    # вертикальные «ряды»
    x=0.0
    xs=[0.0]
    for p in pitches:
        x+=p; xs.append(x)
    for xi in xs:
        ax.axvline(x=xi, ymin=0.05, ymax=0.95, ls=":", lw=0.5)

    # слои: линии и узлы
    for p_idx,p in enumerate(plates):
        nSeg = p.last_row - p.first_row + 1
        x0 = sum(pitches[:p.first_row-1])
        xs_layer = [x0]
        for s in range(nSeg):
            xs_layer.append(xs_layer[-1] + pitches[p.first_row-1+s])
        ys = [y_levels[p_idx]]*(nSeg+1)
        ax.plot(xs_layer, ys, marker="o", lw=2, label=p.name)
        # нагрузки на концах
        if abs(p.Fx_left)>0.0:
            ax.arrow(xs_layer[0]-0.2, ys[0], 0.15, 0.0, head_width=10, head_length=0.15, length_includes_head=True)
            ax.text(xs_layer[0]-0.25, ys[0]+12, f"{p.Fx_left:.0f} lb", ha="right", va="bottom")
        if abs(p.Fx_right)>0.0:
            ax.arrow(xs_layer[-1]+0.2, ys[-1], -0.15, 0.0, head_width=10, head_length=0.15, length_includes_head=True)
            ax.text(xs_layer[-1]+0.25, ys[-1]+12, f"{p.Fx_right:.0f} lb", ha="left", va="bottom")
        # опоры
        for (pi,ln,u0) in supports:
            if pi==p_idx:
                ax.plot(xs_layer[ln], ys[ln]-12, marker= "^", ms=10)

    # пружины крепежа (между соседними слоями)
    for fr in fasteners:
        r=fr.row
        present=[pi for pi,p in enumerate(plates) if p.first_row<=r<=p.last_row]
        present.sort()
        x_r = sum(pitches[:r-1])
        for a,b in zip(present[:-1],present[1:]):
            y1=y_levels[a]; y2=y_levels[b]
            ax.plot([x_r,x_r],[y1,y2], ls="--", lw=1)

    ax.set_xlabel("x [in]")
    ax.set_yticks([y_levels[i] for i in range(len(plates))])
    ax.set_yticklabels([p.name for p in plates])
    ax.legend(loc="upper right")
    ax.set_title("User‑defined scheme (nodes •, supports ▲, fasteners --)")
    st.pyplot(fig)

    # --------- Таблицы ----------
    # Крепеж
    df_fast = pd.DataFrame(out["fasteners"], columns=["Row","Interface","CF [in/lb]","k [lb/in]","F [lb]","iDOF","jDOF"])
    df_fast = df_fast.sort_values(["Row","Interface"])
    # Узлы
    df_nodes = pd.DataFrame(out["node_table"],
                            columns=["plate_id","Plate","local_node","X [in]","u [in]","Net Bypass [lb]","t [in]","Bypass Area [in^2]"])
    # Стержни
    df_bars = pd.DataFrame(out["bar_table"], columns=["plate_id","Plate","seg","Force [lb]","k_bar [lb/in]","E [psi]"])
    # Bearing/Bypass
    df_bb = pd.DataFrame(out["bearing_bypass"], columns=["Row","Plate","Bearing [lb]","Bypass [lb]"])

    st.subheader("Fasteners")
    st.dataframe(df_fast.style.format({"CF [in/lb]":"{:.3e}","k [lb/in]":"{:.3e}","F [lb]":"{:.2f}"}), use_container_width=True)

    st.subheader("Nodes")
    st.dataframe(df_nodes.style.format({"X [in]":"{:.3f}","u [in]":"{:.6e}","Net Bypass [lb]":"{:.2f}","t [in]":"{:.3f}","Bypass Area [in^2]":"{:.3f}"}),
                 use_container_width=True)

    st.subheader("Bars (plate segments)")
    st.dataframe(df_bars.style.format({"Force [lb]":"{:.2f}","k_bar [lb/in]":"{:.3e}","E [psi]":"{:.3e}"}), use_container_width=True)

    st.subheader("Bearing / Bypass by row & plate")
    st.dataframe(df_bb.style.format({"Bearing [lb]":"{:.2f}","Bypass [lb]":"{:.2f}"}), use_container_width=True)

    # Экспорт
    st.download_button("Export fasteners CSV", data=df_fast.to_csv(index=False).encode("utf-8"),
                       file_name="fasteners.csv", mime="text/csv")
    st.download_button("Export nodes CSV", data=df_nodes.to_csv(index=False).encode("utf-8"),
                       file_name="nodes.csv", mime="text/csv")
    st.download_button("Export bars CSV", data=df_bars.to_csv(index=False).encode("utf-8"),
                       file_name="bars.csv", mime="text/csv")
    st.download_button("Export bearing_bypass CSV", data=df_bb.to_csv(index=False).encode("utf-8"),
                       file_name="bearing_bypass.csv", mime="text/csv")

else:
    st.info("Соберите схему в левой панели и нажмите **Solve**. Для воспроизведения вашего скрина нажмите **Load ▶ JOLT Figure 76**.")
