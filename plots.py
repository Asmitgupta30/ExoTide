"""
plots.py  (Module 2 — Tidal Locking)
======================================
Publication-quality plots for the tidal locking and rotation module.
All plots: Times New Roman, white background, black axes (academic style).

Figures:
  1. RM Anomaly plot (RV vs time with analytic fit)
  2. Transmission spectra comparison (ingress vs egress — redshift/blueshift)
  3. Tidal locking sensitivity heatmap (Q vs k2 grid)
  4. Rotation period periodogram
  5. RM posterior corner plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _set_style():
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "axes.edgecolor":   "black",
        "axes.labelcolor":  "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "text.color":       "black",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
        "axes.grid":        True,
        "axes.linewidth":   1.5,
        "font.size":        16,
        "axes.titlesize":   20,
        "axes.labelsize":   18,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  14,
        "legend.title_fontsize": 16,
    })


# ── 1. RM Anomaly Plot ────────────────────────────────────────────────
def plot_rm_anomaly(t_rv, rv_data, rv_err, rv_model, rm_results,
                    t0, target_name, save_path=None):
    """
    Plot RV anomaly data vs. the fitted OTS analytic model.

    Parameters
    ----------
    t_rv      : array  Time array (days or hours)
    rv_data   : array  Observed RV (km/s)
    rv_err    : array  RV errors (km/s)
    rv_model  : array  Predicted RV from best-fit OTS model
    rm_results : dict  Output from RMSolverMCMC.run_mcmc()
    t0        : float  Mid-transit epoch
    target_name : str
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(t_rv, rv_data, yerr=rv_err,
                fmt="o", color="black", markersize=5, alpha=0.65,
                label="Observed RV", zorder=3)

    sort_i = np.argsort(t_rv)
    ax.plot(t_rv[sort_i], rv_model[sort_i],
            color="crimson", linewidth=2.2,
            label="OTS Analytic Fit", zorder=4)

    ax.axvline(t0, color="gray", linestyle="--", alpha=0.7, label="Mid-Transit")

    v_si  = rm_results["v_sin_i"]["median"]
    lam_d = rm_results["lambda_deg"]["median"]
    state = rm_results.get("tidal_state", "")

    textstr = (f"$v\\,\\sin i$ = {v_si:.2f} km/s\n"
               f"$\\lambda$ = {lam_d:.1f}$^\\circ$\n"
               f"{state}")
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="black", alpha=0.9))

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Radial Velocity Anomaly (km/s)", fontsize=12)
    ax.set_title(f"Rossiter-McLaughlin Effect: {target_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=True, edgecolor="black", loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 2. Transmission Spectra — Redshift / Blueshift ────────────────────
def plot_transmission_spectra(spec_data, save_path=None):
    """
    Ingress vs. egress transmission spectra showing atmospheric Doppler shifts.
    This is the direct observational signature of a non-tidally-locked atmosphere.
    """
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    wl = spec_data["wavelength_nm"]
    f_in = spec_data["flux_ingress"]
    f_eg = spec_data["flux_egress"]
    lc   = spec_data["line_center_nm"]
    dl_in = spec_data["dl_ingress_nm"]
    dl_eg = spec_data["dl_egress_nm"]
    dv    = spec_data["delta_v_km_s"]

    # Left panel: both spectra overlaid
    ax = axes[0]
    ax.plot(wl, f_in, color="firebrick",  linewidth=1.8,
            label=f"Ingress (morning limb): v={spec_data['v_ingress_km_s']:+.1f} km/s (redshift)")
    ax.plot(wl, f_eg, color="steelblue",  linewidth=1.8,
            label=f"Egress  (evening limb): v={spec_data['v_egress_km_s']:+.1f} km/s (blueshift)")
    ax.axvline(lc,         color="black",     linestyle=":",  alpha=0.7, label="Rest wavelength")
    ax.axvline(lc + dl_in, color="firebrick", linestyle="--", alpha=0.6, label="Ingress shifted")
    ax.axvline(lc + dl_eg, color="steelblue", linestyle="--", alpha=0.6, label="Egress shifted")
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Normalized Flux", fontsize=12)
    ax.set_title("Ingress vs. Egress Transmission Spectra\n"
                 "(Atmospheric Atmospheric Super-Rotation Signature)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=True, edgecolor="black", fontsize=9)

    # Right panel: difference spectrum
    ax = axes[1]
    diff = f_eg - f_in
    ax.plot(wl, diff, color="black", linewidth=1.4)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(lc, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(wl, diff, where=(diff > 0), alpha=0.3, color="steelblue",
                    label="Egress excess (blueshift)")
    ax.fill_between(wl, diff, where=(diff < 0), alpha=0.3, color="firebrick",
                    label="Ingress excess (redshift)")
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Flux difference (Egress - Ingress)", fontsize=12)
    ax.set_title(f"Difference Spectrum\n"
                 f"$\\Delta v_{{atm}}$ = {dv:.1f} km/s  [Super-rotating jet]",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=True, edgecolor="black", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 3. Tidal Sensitivity Heatmap ─────────────────────────────────────
def plot_sensitivity_heatmap(sens_data, save_path=None):
    """
    Heatmap of age/tau_sync over the Q–k2 parameter space.
    Values > 1 (warm colours) indicate the planet is/was tidally locked.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    grid    = np.log10(np.clip(sens_data["age_to_tau_grid"], 1e-3, 1e4))
    Q_vals  = sens_data["Q_values"]
    k2_vals = sens_data["k2_values"]

    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=-3, vmax=3,
                   extent=[-0.5, len(k2_vals)-0.5, -0.5, len(Q_vals)-0.5])

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("log$_{10}$(Age / $\\tau_{sync}$)", fontsize=11)

    ax.set_xticks(range(len(k2_vals)))
    ax.set_xticklabels([f"{k:.2f}" for k in k2_vals])
    ax.set_yticks(range(len(Q_vals)))
    ax.set_yticklabels([f"{int(q)}" for q in Q_vals])
    ax.set_xlabel("Love number $k_2$", fontsize=12)
    ax.set_ylabel("Tidal dissipation factor $Q$", fontsize=12)
    ax.set_title("Tidal Locking Sensitivity Grid\n"
                 "Green = Locked  |  Red = Not Yet Locked",
                 fontsize=12, fontweight="bold")

    ax.axhline(y=np.searchsorted(Q_vals, 100) - 0.5,
               color="black", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 4. Synthetic RV Dataset Plot ──────────────────────────────────────
def plot_synthetic_rv(rv_data, save_path=None):
    """Full plot of the synthetic non-locked planet RV dataset."""
    _set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    t   = rv_data["time_hrs"]
    rv  = rv_data["rv_km_s"]
    err = rv_data["rv_err_km_s"]
    rv_true = rv_data["rv_true_km_s"]

    # Color by phase
    colors = {"out-of-transit": "gray", "ingress": "firebrick",
              "in-transit": "black", "egress": "steelblue"}
    for label, color in colors.items():
        mask = np.array([l == label for l in rv_data["labels"]])
        if np.any(mask):
            ax.errorbar(t[mask], rv[mask], yerr=err[mask],
                        fmt="o", color=color, markersize=6, alpha=0.8,
                        label=label.replace("-", " ").title(), zorder=3)

    ax.plot(t, rv_true, color="black", linewidth=1.5, linestyle="--",
            alpha=0.7, label="True signal (no noise)", zorder=4)

    t_in   = rv_data["t_ingress"]
    t_eg   = rv_data["t_egress"]
    ax.axvspan(t_in[0], t_in[1], alpha=0.08, color="firebrick")
    ax.axvspan(t_eg[0], t_eg[1], alpha=0.08, color="steelblue")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Mid-transit")

    ax.annotate(f"Ingress: v={rv_data['v_ingress']:+.1f} km/s\n(REDSHIFT)",
                xy=(np.mean(t_in), 0.6), fontsize=9, color="firebrick",
                ha="center", style="italic")
    ax.annotate(f"Egress: v={rv_data['v_egress']:+.1f} km/s\n(BLUESHIFT)",
                xy=(np.mean(t_eg), -0.9), fontsize=9, color="steelblue",
                ha="center", style="italic")

    ax.set_xlabel("Time from Mid-Transit (hours)", fontsize=12)
    ax.set_ylabel("Radial Velocity Anomaly (km/s)", fontsize=12)
    ax.set_title("Synthetic Non-Tidally-Locked Planet — Atmospheric Super-Rotation\n"
                 "(Ingress Redshift + Egress Blueshift = Super-Rotating Equatorial Jet)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=True, edgecolor="black", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 5. RM Posterior Corner ────────────────────────────────────────────
def plot_rm_corner(samples, save_path=None):
    """Corner plot for the 2-parameter RM MCMC [v_sin_i, lambda]."""
    try:
        import corner
    except ImportError:
        print("  corner not installed. pip install corner")
        return None
    _set_style()
    labels = [r"$v\,\sin i$ (km/s)", r"$\lambda$ (rad)"]
    fig = corner.corner(
        samples, labels=labels,
        show_titles=True, title_kwargs={"fontsize": 10},
        color="black",
        hist_kwargs={"color": "black", "linewidth": 1.2},
        quantiles=[0.16, 0.5, 0.84],
        plot_contours=True,
        contour_kwargs={"colors": "black", "linewidths": 1.0},
    )
    fig.suptitle("RM Effect MCMC Posteriors ($v\\,\\sin i$ and $\\lambda$)",
                 fontsize=12, fontweight="bold", y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 6. Tidal Evolution (tau_sync vs a) ────────────────────────────────
def plot_tidal_evolution(a_range, tau_by_type, sys_age, planet_a, save_path=None):
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"rocky": "saddlebrown", "ice_giant": "steelblue", "gas_giant": "darkorange"}
    labels = {"rocky": "Rocky ($Q=10^2$)", "ice_giant": "Ice Giant ($Q=10^3$)", "gas_giant": "Gas Giant ($Q=10^5$)"}

    for ptype, tau in tau_by_type.items():
        ax.plot(a_range, tau, color=colors[ptype], linewidth=3.0, label=labels[ptype])

    ax.axhline(sys_age, color="black", linestyle="--", linewidth=2.0, label=f"System Age ({sys_age} Gyr)")
    ax.axvline(planet_a, color="crimson", linestyle=":", linewidth=2.5, label=f"Planet $a$ = {planet_a:.3f} AU")

    ax.fill_between(a_range, 1e-5, sys_age, alpha=0.1, color="green", label="Tidally Locked")
    ax.fill_between(a_range, sys_age, 1e12, alpha=0.1, color="red", label="Not Locked")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Semi-major Axis $a$ (AU)", fontsize=18)
    ax.set_ylabel(r"Synchronization Time $\tau_{sync}$ (Gyr)", fontsize=18)
    ax.set_title("Tidal Evolution vs. Orbital Distance", fontsize=20, fontweight="bold")
    ax.set_xlim(a_range[0], a_range[-1])
    ax.set_ylim(1e-4, 1e8)
    
    ax.legend(frameon=True, edgecolor="black", fontsize=14, loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 7. Pseudo-sync vs Eccentricity ────────────────────────────────────
def plot_pseudo_sync_ecc(ecc_range, ps_ratios, planet_ecc, save_path=None):
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ecc_range, ps_ratios, color="indigo", linewidth=3.0, label=r"$\omega_{ps} / n$ (Hut 1981)")
    
    ax.axvline(planet_ecc, color="crimson", linestyle=":", linewidth=2.5, label=f"Planet $e$ = {planet_ecc:.3f}")
    
    ps_val = ps_ratios[np.argmin(np.abs(ecc_range - planet_ecc))]
    ax.axhline(ps_val, color="gray", linestyle="--", linewidth=2.0)
    ax.plot(planet_ecc, ps_val, marker="o", color="crimson", markersize=8)

    ax.set_xlabel("Eccentricity $e$", fontsize=18)
    ax.set_ylabel(r"Pseudo-sync Ratio $\omega_{ps} / n$", fontsize=18)
    ax.set_title("Pseudo-synchronous Rotation Rate", fontsize=20, fontweight="bold")
    ax.set_xlim(0, 0.85)
    
    ax.legend(frameon=True, edgecolor="black", fontsize=14, loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 8. Full Sensitivity Dashboard ─────────────────────────────────────
def plot_sensitivity_dashboard(sens, save_path=None):
    _set_style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    
    sys_age = sens["system_age_gyr"]
    
    # Panel 1: tau vs a
    ax = axes[0, 0]
    colors = {"rocky": "saddlebrown", "ice_giant": "steelblue", "gas_giant": "darkorange"}
    for ptype, tau in sens["tau_by_type"].items():
        ax.plot(sens["a_range"], tau, color=colors[ptype], linewidth=3.0, label=ptype.replace('_', ' ').title())
    ax.axhline(sys_age, color="black", linestyle="--", linewidth=2.0, label="System Age")
    ax.axvline(sens["planet_a_AU"], color="crimson", linestyle=":", linewidth=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Semi-major Axis $a$ (AU)", fontsize=18)
    ax.set_ylabel(r"$\tau_{sync}$ (Gyr)", fontsize=18)
    ax.set_title("Sensitivity to Distance", fontsize=20, fontweight="bold")
    ax.legend(fontsize=14)

    # Panel 2: tau vs R_planet
    ax = axes[0, 1]
    ax.plot(sens["R_range"], sens["tau_vs_R"], color="saddlebrown", linewidth=3.0)
    ax.axhline(sys_age, color="black", linestyle="--", linewidth=2.0)
    ax.axvline(sens["planet_R"], color="crimson", linestyle=":", linewidth=2.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Planet Radius $R_\oplus$", fontsize=18)
    ax.set_ylabel(r"$\tau_{sync}$ (Gyr)", fontsize=18)
    ax.set_title("Sensitivity to Planet Size", fontsize=20, fontweight="bold")

    # Panel 3: tau vs M_star
    ax = axes[0, 2]
    ax.plot(sens["Mstar_range"], sens["tau_vs_Mstar"], color="darkorange", linewidth=3.0)
    ax.axhline(sys_age, color="black", linestyle="--", linewidth=2.0)
    ax.axvline(sens["planet_Mstar"], color="crimson", linestyle=":", linewidth=2.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Stellar Mass $M_\odot$", fontsize=18)
    ax.set_ylabel(r"$\tau_{sync}$ (Gyr)", fontsize=18)
    ax.set_title("Sensitivity to Stellar Mass", fontsize=20, fontweight="bold")

    # Panel 4: tau vs Q (for different k2)
    ax = axes[1, 0]
    styles = ["-", "--", ":"]
    for i, k2 in enumerate(sens["k2_vals_sens"]):
        ax.plot(sens["Q_range"], sens["tau_vs_Q"][k2], color="teal", linestyle=styles[i], linewidth=3.0, label=f"$k_2$={k2}")
    ax.axhline(sys_age, color="black", linestyle="--", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tidal Dissipation $Q$", fontsize=18)
    ax.set_ylabel(r"$\tau_{sync}$ (Gyr)", fontsize=18)
    ax.set_title("Sensitivity to Tidal Parameters", fontsize=20, fontweight="bold")
    ax.legend(fontsize=14)

    # Panel 5: Pseudo-sync
    ax = axes[1, 1]
    ax.plot(sens["ecc_range"], sens["ps_ratios"], color="indigo", linewidth=3.0)
    ax.axvline(sens["planet_ecc"], color="crimson", linestyle=":", linewidth=2.5)
    ax.set_xlabel("Eccentricity $e$", fontsize=18)
    ax.set_ylabel(r"$\omega_{ps} / n$", fontsize=18)
    ax.set_title("Pseudo-synchronous Rate", fontsize=20, fontweight="bold")

    # Panel 6: Q-k2 Heatmap
    ax = axes[1, 2]
    grid = np.log10(np.clip(sens["sens_grid"]["age_to_tau_grid"], 1e-3, 1e4))
    Q_vals = sens["sens_grid"]["Q_values"]
    k2_vals = sens["sens_grid"]["k2_values"]
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-3, vmax=3,
                   extent=[-0.5, len(k2_vals)-0.5, -0.5, len(Q_vals)-0.5])
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r"$\log_{10}(\mathrm{Age} / \tau_{sync})$", fontsize=16)
    ax.set_xticks(range(len(k2_vals)))
    ax.set_xticklabels([f"{k:.2f}" for k in k2_vals])
    ax.set_yticks(range(len(Q_vals)))
    ax.set_yticklabels([f"{int(q)}" for q in Q_vals])
    ax.set_xlabel(r"Love number $k_2$", fontsize=18)
    ax.set_ylabel(r"Tidal Dissipation $Q$", fontsize=18)
    ax.set_title("Q-$k_2$ Locking Grid", fontsize=20, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig
