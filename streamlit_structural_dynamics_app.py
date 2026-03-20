import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Structural Dynamics Explorer", layout="wide")

st.title("Structural Dynamics Explorer")
st.caption("For SDOF and MDOF systems: natural frequencies, periods, mode shapes, and static total deformation under load.")


def safe_float_list(text, n=None, name="values"):
    vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    if n is not None and len(vals) != n:
        raise ValueError(f"Expected {n} {name}, but got {len(vals)}.")
    return vals


def influence_vector(n):
    return np.ones((n, 1))


def build_shear_building_matrices(masses, stiffnesses):
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            if n == 1:
                K[i, i] = stiffnesses[0]
            else:
                K[i, i] = stiffnesses[0] + stiffnesses[1]
                K[i, i + 1] = -stiffnesses[1]
        elif i == n - 1:
            K[i, i] = stiffnesses[i]
            K[i, i - 1] = -stiffnesses[i]
        else:
            K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
            K[i, i - 1] = -stiffnesses[i]
            K[i, i + 1] = -stiffnesses[i + 1]
    for i in range(1, n):
        K[i - 1, i] = K[i, i - 1]
    return M, K


def modal_analysis(M, K):
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    omegas = np.sqrt(np.clip(eigvals, 0, None))
    freqs = omegas / (2 * np.pi)
    periods = np.array([np.inf if w == 0 else 2 * np.pi / w for w in omegas])

    # Mass-normalize mode shapes
    phi = eigvecs.copy()
    for i in range(phi.shape[1]):
        norm = np.sqrt(phi[:, i].T @ M @ phi[:, i])
        if norm != 0:
            phi[:, i] = phi[:, i] / norm
        # fix sign for cleaner plotting
        if phi[np.argmax(np.abs(phi[:, i])), i] < 0:
            phi[:, i] *= -1

    return eigvals, omegas, freqs, periods, phi


def static_displacement(K, F):
    return np.linalg.solve(K, F)


def participation_factors(M, phi):
    r = influence_vector(M.shape[0])
    gammas = []
    eff_mass = []
    total_mass = float(r.T @ M @ r)
    for i in range(phi.shape[1]):
        p = phi[:, [i]]
        num = float(p.T @ M @ r)
        den = float(p.T @ M @ p)
        gamma = num / den if den != 0 else np.nan
        meff = (num ** 2) / den if den != 0 else np.nan
        gammas.append(gamma)
        eff_mass.append(meff / total_mass * 100 if total_mass != 0 else np.nan)
    return np.array(gammas), np.array(eff_mass)


def plot_mode_shape(phi_col, title):
    n = len(phi_col)
    y = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0] + list(phi_col), [0] + list(y), marker="o")
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Relative modal amplitude")
    ax.set_ylabel("DOF / Storey")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_deformation(u, title):
    n = len(u)
    y = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0] + list(u.flatten()), [0] + list(y), marker="o")
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Static displacement")
    ax.set_ylabel("DOF / Storey")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


with st.sidebar:
    st.header("Model Setup")
    system_type = st.radio("Choose system", ["SDOF", "MDOF shear building"], index=1)
    units = st.selectbox("Units note", ["SI (N, kg, m)", "kN, tonne, m", "Custom consistent units"])
    st.caption("Use consistent units. For example, if stiffness is kN/m, loads should be kN and mass consistent with that unit system.")

if system_type == "SDOF":
    st.subheader("Single Degree of Freedom (SDOF)")
    col1, col2, col3 = st.columns(3)
    with col1:
        m = st.number_input("Mass, m", min_value=0.0001, value=1000.0, step=100.0)
    with col2:
        k = st.number_input("Stiffness, k", min_value=0.0001, value=20000.0, step=1000.0)
    with col3:
        F = st.number_input("Applied static load, F", value=10.0, step=1.0)

    omega = math.sqrt(k / m)
    freq = omega / (2 * math.pi)
    period = 2 * math.pi / omega
    u = F / k

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ωₙ (rad/s)", f"{omega:.4f}")
    c2.metric("fₙ (Hz)", f"{freq:.4f}")
    c3.metric("Tₙ (s)", f"{period:.4f}")
    c4.metric("Static deformation", f"{u:.6f}")

    st.markdown("### Governing relations")
    st.latex(r"\omega_n=\sqrt{\frac{k}{m}},\quad f_n=\frac{\omega_n}{2\pi},\quad T_n=\frac{2\pi}{\omega_n},\quad u=\frac{F}{k}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["DOF 1"], [u])
    ax.set_ylabel("Displacement")
    ax.set_title("Static deformation under applied load")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

else:
    st.subheader("Multi Degree of Freedom (MDOF) Shear Building")
    st.write("This version assumes a lumped-mass shear-building model with one lateral DOF per floor.")

    n = st.number_input("Number of DOF / storeys", min_value=2, max_value=10, value=3, step=1)

    default_masses = ", ".join(["1000"] * int(n))
    default_stiff = ", ".join(["20000"] * int(n))
    default_loads = ", ".join(["10"] * int(n))

    masses_text = st.text_input(f"Masses m1..m{n} (comma-separated)", value=default_masses)
    stiffness_text = st.text_input(f"Storey stiffnesses k1..k{n} (comma-separated)", value=default_stiff)
    load_text = st.text_input(f"Lateral loads F1..F{n} (comma-separated)", value=default_loads)

    try:
        masses = np.array(safe_float_list(masses_text, int(n), "masses"), dtype=float)
        stiffnesses = np.array(safe_float_list(stiffness_text, int(n), "stiffnesses"), dtype=float)
        loads = np.array(safe_float_list(load_text, int(n), "loads"), dtype=float).reshape(-1, 1)

        M, K = build_shear_building_matrices(masses, stiffnesses)
        eigvals, omegas, freqs, periods, phi = modal_analysis(M, K)
        u = static_displacement(K, loads)
        gammas, eff_mass_pct = participation_factors(M, phi)

        st.markdown("### Global matrices")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Mass matrix M**")
            st.dataframe(pd.DataFrame(M))
        with c2:
            st.write("**Stiffness matrix K**")
            st.dataframe(pd.DataFrame(K))

        results = pd.DataFrame({
            "Mode": np.arange(1, int(n) + 1),
            "Eigenvalue (ω²)": eigvals,
            "ω (rad/s)": omegas,
            "f (Hz)": freqs,
            "T (s)": periods,
            "Participation factor Γ": gammas,
            "Effective modal mass (%)": eff_mass_pct,
        })
        st.markdown("### Modal properties")
        st.dataframe(results, use_container_width=True)

        st.markdown("### Mode shapes")
        mode_cols = st.columns(min(3, int(n)))
        for i in range(min(3, int(n))):
            with mode_cols[i]:
                st.pyplot(plot_mode_shape(phi[:, i], f"Mode {i+1}"))

        if int(n) > 3:
            with st.expander("Show remaining mode-shape plots"):
                rem_cols = st.columns(2)
                for i in range(3, int(n)):
                    with rem_cols[(i - 3) % 2]:
                        st.pyplot(plot_mode_shape(phi[:, i], f"Mode {i+1}"))

        st.markdown("### Modal shape values")
        mode_shape_df = pd.DataFrame(phi, columns=[f"Mode {i+1}" for i in range(int(n))])
        mode_shape_df.index = [f"DOF {i+1}" for i in range(int(n))]
        st.dataframe(mode_shape_df, use_container_width=True)

        st.markdown("### Static total deformation from applied load")
        st.write("Load vector:", pd.DataFrame(loads, index=[f"DOF {i+1}" for i in range(int(n))], columns=["Load"]))
        disp_df = pd.DataFrame(u, index=[f"DOF {i+1}" for i in range(int(n))], columns=["Displacement"])
        st.dataframe(disp_df, use_container_width=True)
        st.pyplot(plot_deformation(u, "Static deformed shape"))

        st.markdown("### Notes for class discussion")
        st.info(
            "This app performs linear elastic modal analysis for a lumped-mass shear building. "
            "Mode shapes are mass-normalized. Static deformation is solved from Ku = F."
        )

    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

st.markdown("---")
st.caption("Built for classroom demonstration. You can extend this to include damping, response history, modal superposition, and response spectrum analysis.")
