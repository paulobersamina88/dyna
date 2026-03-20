import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Structural Dynamics Teaching App", layout="wide")

st.title("Structural Dynamics Teaching App")
st.write("SDOF and MDOF systems with harmonic excitation, resonance plots, mode shapes, and deformation response.")

# =========================================================
# Helpers
# =========================================================
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

def harmonic_sdof_amplitude(m, c, k, p0, w):
    denom = np.sqrt((k - m * w**2)**2 + (c * w)**2)
    return p0 / denom

def harmonic_sdof_phase(m, c, k, w):
    return np.arctan2(c * w, (k - m * w**2))

def sdof_time_history(m, c, k, p0, w, t):
    X = harmonic_sdof_amplitude(m, c, k, p0, w)
    phi = harmonic_sdof_phase(m, c, k, w)
    x = X * np.sin(w * t - phi)
    force = p0 * np.sin(w * t)
    return x, force, X, phi

def dynamic_magnification(r, zeta):
    return 1.0 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)

def mdof_matrices(m, k):
    n = len(m)
    M = np.diag(m)
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

    return M, K

def modal_properties(M, K):
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
    return eigvals, wn, fn, Tn, phi, gamma, eff_mass

def rayleigh_damping(alpha, beta, M, K):
    return alpha * M + beta * K

def mdof_harmonic_response(M, C, K, F0, w):
    n = M.shape[0]
    jw = 1j * w
    Z = K - (w**2) * M + jw * C
    X = np.linalg.solve(Z, F0.astype(complex))
    return X

def mdof_frequency_sweep(M, C, K, F0, w_values):
    n = M.shape[0]
    amplitudes = np.zeros((n, len(w_values)))
    for i, w in enumerate(w_values):
        X = mdof_harmonic_response(M, C, K, F0, w)
        amplitudes[:, i] = np.abs(X.flatten())
    return amplitudes

def mdof_time_history_steady_state(M, C, K, F0, w, t):
    X = mdof_harmonic_response(M, C, K, F0, w)
    n = M.shape[0]
    x_t = np.zeros((n, len(t)))
    for i in range(n):
        amp = np.abs(X[i, 0])
        phase = np.angle(X[i, 0])
        x_t[i, :] = amp * np.sin(w * t + phase)
    f_t = np.outer(F0.flatten(), np.sin(w * t))
    return x_t, f_t, X

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Model Setup")
system_type = st.sidebar.radio("Choose system", ["SDOF", "MDOF shear building"])
units_note = st.sidebar.selectbox(
    "Units note",
    ["SI (N, kg, m)", "kN-ton-m (consistent units)", "Custom consistent units"]
)
st.sidebar.write("Use consistent units throughout the model.")

# =========================================================
# SDOF
# =========================================================
if system_type == "SDOF":
    st.header("Single Degree of Freedom (SDOF) with Harmonic Excitation")

    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Mass m", min_value=0.0001, value=1000.0, step=100.0)
        k = st.number_input("Stiffness k", min_value=0.0001, value=20000.0, step=100.0)
        zeta = st.number_input("Damping ratio ζ", min_value=0.0, value=0.05, step=0.01, format="%.4f")
        p0 = st.number_input("Harmonic force amplitude P0", min_value=0.0, value=10.0, step=1.0)
    with col2:
        force_freq_hz = st.number_input("Excitation frequency f (Hz)", min_value=0.0001, value=0.60, step=0.01, format="%.4f")
        time_duration = st.number_input("Time duration for plot (s)", min_value=1.0, value=30.0, step=1.0)
        n_points = st.number_input("Number of time points", min_value=200, value=2000, step=100)

    wn = np.sqrt(k / m)
    fn = wn / (2 * np.pi)
    Tn = 1 / fn
    c = 2 * zeta * m * wn
    w = 2 * np.pi * force_freq_hz
    r = w / wn
    k_static_disp = p0 / k if k != 0 else np.nan

    t = np.linspace(0, time_duration, int(n_points))
    x, force_t, X, phi = sdof_time_history(m, c, k, p0, w, t)
    DMF = X / k_static_disp if k_static_disp != 0 else np.nan

    st.subheader("Key Results")
    result_df = pd.DataFrame({
        "Quantity": [
            "Natural circular frequency ωn (rad/s)",
            "Natural frequency fn (Hz)",
            "Natural period Tn (s)",
            "Damping coefficient c",
            "Excitation circular frequency ω (rad/s)",
            "Frequency ratio r = ω/ωn",
            "Static displacement P0/k",
            "Steady-state amplitude X",
            "Dynamic magnification factor X/(P0/k)",
            "Phase angle φ (deg)"
        ],
        "Value": [
            wn, fn, Tn, c, w, r, k_static_disp, X, DMF, np.degrees(phi)
        ]
    })
    st.dataframe(result_df, use_container_width=True)

    st.subheader("SDOF Time History")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, x)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Displacement")
    ax1.set_title("Steady-State Displacement Response")
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, force_t)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force")
    ax2.set_title("Applied Harmonic Force")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Resonance / Frequency Sweep")
    freq_min = st.number_input("Sweep minimum frequency (Hz)", min_value=0.001, value=0.01, step=0.01, format="%.4f")
    freq_max = st.number_input("Sweep maximum frequency (Hz)", min_value=0.01, value=max(2.5 * fn, 1.5), step=0.05, format="%.4f")
    n_sweep = st.number_input("Number of sweep points", min_value=50, value=400, step=50)

    freqs = np.linspace(freq_min, freq_max, int(n_sweep))
    omega_vals = 2 * np.pi * freqs
    amp_vals = np.array([harmonic_sdof_amplitude(m, c, k, p0, om) for om in omega_vals])
    r_vals = omega_vals / wn
    dmf_vals = np.array([dynamic_magnification(rv, zeta) for rv in r_vals])

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(freqs, amp_vals)
    ax3.axvline(fn, linestyle="--")
    ax3.set_xlabel("Excitation frequency (Hz)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Steady-State Amplitude vs Excitation Frequency")
    ax3.grid(True)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(r_vals, dmf_vals)
    ax4.axvline(1.0, linestyle="--")
    ax4.set_xlabel("Frequency ratio r = ω/ωn")
    ax4.set_ylabel("Dynamic Magnification Factor")
    ax4.set_title("Dynamic Magnification Curve")
    ax4.grid(True)
    st.pyplot(fig4)

# =========================================================
# MDOF
# =========================================================
else:
    st.header("MDOF Shear Building with Harmonic Excitation")
    st.write("Assumes one lateral DOF per floor and harmonic load vector F(t) = F0 sin(ωt).")

    n = st.number_input("Number of DOF / storeys", min_value=2, max_value=20, value=3, step=1)

    masses_text = st.text_input(f"Masses m1..m{n} (comma-separated)", value=", ".join(["1000"] * n))
    stiffness_text = st.text_input(f"Storey stiffnesses k1..k{n} (comma-separated)", value=", ".join(["20000"] * n))
    loads_text = st.text_input(f"Harmonic load amplitudes F1..F{n} (comma-separated)", value=", ".join(["10"] * n))

    st.subheader("Damping")
    damping_model = st.radio("Select damping model", ["Modal damping ratio (uniform, used through Rayleigh fit)", "Direct Rayleigh coefficients α and β"])

    try:
        m = parse_vector(masses_text, n)
        k = parse_vector(stiffness_text, n)
        F0 = parse_vector(loads_text, n).reshape(-1, 1)

        M, K = mdof_matrices(m, k)
        eigvals, wn, fn, Tn, phi, gamma, eff_mass = modal_properties(M, K)

        if damping_model == "Modal damping ratio (uniform, used through Rayleigh fit)":
            zeta = st.number_input("Modal damping ratio ζ", min_value=0.0, value=0.05, step=0.01, format="%.4f")
            if n >= 2:
                w1 = wn[0]
                w2 = wn[1]
            else:
                w1 = wn[0]
                w2 = wn[0] * 2.0

            A_fit = np.array([
                [1 / (2 * w1), w1 / 2],
                [1 / (2 * w2), w2 / 2]
            ])
            b_fit = np.array([zeta, zeta])
            alpha, beta = np.linalg.solve(A_fit, b_fit)
        else:
            alpha = st.number_input("Rayleigh alpha α", value=0.0, step=0.001, format="%.6f")
            beta = st.number_input("Rayleigh beta β", value=0.001, step=0.0001, format="%.6f")

        C = rayleigh_damping(alpha, beta, M, K)

        st.subheader("Matrices")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Mass Matrix [M]**")
            st.dataframe(pd.DataFrame(M), use_container_width=True)
        with col2:
            st.write("**Stiffness Matrix [K]**")
            st.dataframe(pd.DataFrame(K), use_container_width=True)
        with col3:
            st.write("**Damping Matrix [C]**")
            st.dataframe(pd.DataFrame(C), use_container_width=True)

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

        st.subheader("Static Deformation Under Load Amplitudes")
        u_static = np.linalg.solve(K, F0)
        static_df = pd.DataFrame({
            "Floor": np.arange(1, n + 1),
            "Load amplitude": F0.flatten(),
            "Static displacement": u_static.flatten()
        })
        st.dataframe(static_df, use_container_width=True)

        fig_static, ax_static = plt.subplots(figsize=(5, 6))
        ax_static.plot(u_static.flatten(), np.arange(1, n + 1), marker="o")
        ax_static.axvline(0, linewidth=1)
        ax_static.set_xlabel("Displacement")
        ax_static.set_ylabel("Floor")
        ax_static.set_title("Static Deformation Shape")
        ax_static.grid(True)
        st.pyplot(fig_static)

        st.subheader("Single Harmonic Excitation Response")
        excite_freq_hz = st.number_input("Excitation frequency f (Hz)", min_value=0.0001, value=float(fn[0]) if len(fn) > 0 else 0.5, step=0.01, format="%.4f")
        tmax = st.number_input("Time duration for harmonic response (s)", min_value=1.0, value=30.0, step=1.0)
        nt = st.number_input("Number of time points for harmonic response", min_value=200, value=2000, step=100)

        w_exc = 2 * np.pi * excite_freq_hz
        t = np.linspace(0, tmax, int(nt))
        x_t, f_t, X_complex = mdof_time_history_steady_state(M, C, K, F0, w_exc, t)

        amp_df = pd.DataFrame({
            "Floor": np.arange(1, n + 1),
            "Amplitude": np.abs(X_complex.flatten()),
            "Phase (deg)": np.degrees(np.angle(X_complex.flatten()))
        })
        st.dataframe(amp_df, use_container_width=True)

        selected_floor = st.selectbox("Select floor for time-history plot", options=list(range(1, n + 1)), index=n - 1)

        fig_th, ax_th = plt.subplots(figsize=(10, 4))
        ax_th.plot(t, x_t[selected_floor - 1, :])
        ax_th.set_xlabel("Time (s)")
        ax_th.set_ylabel("Displacement")
        ax_th.set_title(f"Steady-State Displacement Response - Floor {selected_floor}")
        ax_th.grid(True)
        st.pyplot(fig_th)

        st.subheader("Frequency Sweep / Resonance")
        sweep_min = st.number_input("Sweep minimum frequency (Hz)", min_value=0.001, value=0.01, step=0.01, format="%.4f")
        sweep_max_default = max(float(fn[-1] * 2.5), 1.5)
        sweep_max = st.number_input("Sweep maximum frequency (Hz)", min_value=0.01, value=sweep_max_default, step=0.05, format="%.4f")
        sweep_pts = st.number_input("Number of sweep points", min_value=50, value=400, step=50)

        sweep_freqs = np.linspace(sweep_min, sweep_max, int(sweep_pts))
        sweep_omega = 2 * np.pi * sweep_freqs
        amps = mdof_frequency_sweep(M, C, K, F0, sweep_omega)

        floor_to_plot = st.selectbox("Select floor for resonance plot", options=list(range(1, n + 1)), index=n - 1, key="floor_resonance")

        fig_res, ax_res = plt.subplots(figsize=(10, 5))
        ax_res.plot(sweep_freqs, amps[floor_to_plot - 1, :], label=f"Floor {floor_to_plot}")
        for natf in fn:
            ax_res.axvline(natf, linestyle="--")
        ax_res.set_xlabel("Excitation frequency (Hz)")
        ax_res.set_ylabel("Displacement amplitude")
        ax_res.set_title(f"Resonance Curve - Floor {floor_to_plot}")
        ax_res.grid(True)
        ax_res.legend()
        st.pyplot(fig_res)

        st.subheader("All Floors Resonance Curves")
        fig_all, ax_all = plt.subplots(figsize=(10, 5))
        for i in range(n):
            ax_all.plot(sweep_freqs, amps[i, :], label=f"Floor {i+1}")
        for natf in fn:
            ax_all.axvline(natf, linestyle="--")
        ax_all.set_xlabel("Excitation frequency (Hz)")
        ax_all.set_ylabel("Displacement amplitude")
        ax_all.set_title("Frequency Response of All Floors")
        ax_all.grid(True)
        ax_all.legend()
        st.pyplot(fig_all)

    except Exception as e:
        st.error(f"Input error: {e}")
