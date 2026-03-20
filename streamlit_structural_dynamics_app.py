import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Structural Dynamics Class App", layout="wide")

st.title("Structural Dynamics Teaching App")
st.write("For SDOF and MDOF systems: frequencies, periods, mode shapes, and total deformation under load.")

# -----------------------------
# Helper functions
# -----------------------------
def parse_vector(text, expected_len=None):
    vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    arr = np.array(vals, dtype=float)
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"Expected {expected_len} values, but got {len(arr)}.")
    return arr

def influence_vector(n):
    return np.ones((n, 1))

def normalize_mode_shapes(phi):
    phi_norm = phi.copy().astype(float)
    for i in range(phi_norm.shape[1]):
        max_abs = np.max(np.abs(phi_norm[:, i]))
        if max_abs != 0:
            phi_norm[:, i] = phi_norm[:, i] / max_abs
    return phi_norm

def participation_factors(M, phi):
    r = influence_vector(M.shape[0])
    gammas = []
    eff_mass = []

    total_mass = (r.T @ M @ r).item()

    for i in range(phi.shape[1]):
        p = phi[:, [i]]
        num = (p.T @ M @ r).item()
        den = (p.T @ M @ p).item()

        gamma = num / den if den != 0 else np.nan
        meff = (num ** 2) / den if den != 0 else np.nan

        gammas.append(gamma)
        eff_mass.append(meff / total_mass * 100 if total_mass != 0 else np.nan)

    return np.array(gammas), np.array(eff_mass)

def plot_mode_shape(phi_mode, title):
    floors = np.arange(1, len(phi_mode) + 1)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(phi_mode, floors, marker="o")
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Relative displacement")
    ax.set_ylabel("Floor")
    ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Model Setup")
system_type = st.sidebar.radio("Choose system", ["SDOF", "MDOF shear building"])
units_note = st.sidebar.selectbox("Units note", ["SI (N, kg, m)", "kN-ton-m (consistent units)", "Custom consistent units"])

st.sidebar.write("Use consistent units. Example: if stiffness is in kN/m, loads should be in kN and mass should be consistent with that system.")

# -----------------------------
# SDOF
# -----------------------------
if system_type == "SDOF":
    st.header("Single Degree of Freedom (SDOF)")

    m = st.number_input("Mass m", min_value=0.0001, value=1000.0, step=100.0)
    k = st.number_input("Stiffness k", min_value=0.0001, value=20000.0, step=100.0)
    p = st.number_input("Applied load P", value=10.0, step=1.0)

    if m > 0 and k > 0:
        wn = np.sqrt(k / m)
        fn = wn / (2 * np.pi)
        Tn = 1 / fn
        delta = p / k

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Results")
            st.write(f"Natural circular frequency, ωₙ = **{wn:.6f} rad/s**")
            st.write(f"Natural frequency, fₙ = **{fn:.6f} Hz**")
            st.write(f"Period, Tₙ = **{Tn:.6f} s**")
            st.write(f"Static deformation, Δ = P/k = **{delta:.6f}**")

        with col2:
            result_df = pd.DataFrame({
                "Quantity": ["Mass", "Stiffness", "Load", "ωₙ", "fₙ", "Tₙ", "Δ"],
                "Value": [m, k, p, wn, fn, Tn, delta]
            })
            st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot([0, delta], [0, 0], marker="o")
        ax.set_title("SDOF Static Deformation")
        ax.set_xlabel("Displacement")
        ax.set_yticks([])
        ax.grid(True)
        st.pyplot(fig)

# -----------------------------
# MDOF
# -----------------------------
else:
    st.header("Multi Degree of Freedom (MDOF) Shear Building")
    st.write("This version assumes a lumped-mass shear-building model with one lateral DOF per floor.")

    n = st.number_input("Number of DOF / storeys", min_value=2, max_value=20, value=3, step=1)

    masses_text = st.text_input(f"Masses m1..m{n} (comma-separated)", value=", ".join(["1000"] * n))
    stiffness_text = st.text_input(f"Storey stiffnesses k1..k{n} (comma-separated)", value=", ".join(["20000"] * n))
    loads_text = st.text_input(f"Lateral loads F1..F{n} (comma-separated)", value=", ".join(["10"] * n))

    try:
        m = parse_vector(masses_text, n)
        k = parse_vector(stiffness_text, n)
        F = parse_vector(loads_text, n).reshape(-1, 1)

        # Mass matrix
        M = np.diag(m)

        # Stiffness matrix for shear building
        K = np.zeros((n, n), dtype=float)

        for i in range(n):
            if i == 0:
                if n == 1:
                    K[i, i] = k[i]
                else:
                    K[i, i] = k[i] + k[i + 1]
                    K[i, i + 1] = -k[i + 1]
            elif i == n - 1:
                K[i, i] = k[i]
                K[i, i - 1] = -k[i]
            else:
                K[i, i] = k[i] + k[i + 1]
                K[i, i - 1] = -k[i]
                K[i, i + 1] = -k[i + 1]

        # Static deformation
        u = np.linalg.solve(K, F)

        # Modal analysis
        A = np.linalg.inv(M) @ K
        eigvals, eigvecs = np.linalg.eig(A)

        idx = np.argsort(eigvals.real)
        eigvals = eigvals[idx].real
        eigvecs = eigvecs[:, idx].real

        wn = np.sqrt(np.maximum(eigvals, 0))
        fn = wn / (2 * np.pi)
        Tn = np.where(fn > 0, 1 / fn, np.nan)

        phi = normalize_mode_shapes(eigvecs)
        gamma, eff_mass = participation_factors(M, phi)

        st.subheader("Matrices")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mass Matrix [M]**")
            st.dataframe(pd.DataFrame(M), use_container_width=True)
        with col2:
            st.write("**Stiffness Matrix [K]**")
            st.dataframe(pd.DataFrame(K), use_container_width=True)

        st.subheader("Modal Properties")
        modal_df = pd.DataFrame({
            "Mode": np.arange(1, n + 1),
            "Eigenvalue λ": eigvals,
            "ωn (rad/s)": wn,
            "fn (Hz)": fn,
            "Tn (s)": Tn,
            "Participation Factor Γ": gamma,
            "Effective Modal Mass (%)": eff_mass
        })
        st.dataframe(modal_df, use_container_width=True)

        st.subheader("Mode Shapes")
        mode_shape_df = pd.DataFrame(phi, columns=[f"Mode {i}" for i in range(1, n + 1)])
        mode_shape_df.index = [f"Floor {i}" for i in range(1, n + 1)]
        st.dataframe(mode_shape_df, use_container_width=True)

        for i in range(n):
            plot_mode_shape(phi[:, i], f"Mode Shape {i+1}")

        st.subheader("Static Deformation Under Applied Load")
        deform_df = pd.DataFrame({
            "Floor": np.arange(1, n + 1),
            "Load": F.flatten(),
            "Displacement": u.flatten()
        })
        st.dataframe(deform_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(u.flatten(), np.arange(1, n + 1), marker="o")
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Displacement")
        ax.set_ylabel("Floor")
        ax.set_title("Total Static Deformation Shape")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Input error: {e}")
