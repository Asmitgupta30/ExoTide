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
import os


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
    is_locked = spec_data.get("is_locked", False)
    if is_locked:
        ax.set_title("Ingress vs. Egress Transmission Spectra\n"
                     "(Tidally Locked: Lines Overlap — Co-Rotating Atmosphere)",
                     fontsize=11, fontweight="bold")
    else:
        ax.set_title("Ingress vs. Egress Transmission Spectra\n"
                     "(Free Rotator: Lines Shifted — Atmospheric Super-Rotation)",
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
    is_locked = spec_data.get("is_locked", False)
    if is_locked:
        ax.set_title(f"Difference Spectrum\n"
                     f"$\\Delta v_{{atm}}$ = {dv:.2f} km/s  [Symmetric — Locked]",
                     fontsize=11, fontweight="bold")
    else:
        ax.set_title(f"Difference Spectrum\n"
                     f"$\\Delta v_{{atm}}$ = {dv:.1f} km/s  [Super-Rotating Jet]",
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
    is_locked = rv_data.get("is_locked", False)
    if is_locked:
        ax.set_title("Synthetic Tidally Locked Planet — RM Effect (Symmetric)\n"
                     "(No net atmospheric Doppler shift: co-rotating atmosphere)",
                     fontsize=11, fontweight="bold")
    else:
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


# ── 9. Batch Locking Map ──────────────────────────────────────────────
def plot_batch_locking_map(df, boundary=None, save_path=None):
    """
    Scatter plot of all HWO planets in (a, R_p) space coloured by
    log10(age/tau_sync).  The empirical locking boundary is overplotted
    if supplied.

    Parameters
    ----------
    df       : pd.DataFrame   Output from BatchTidalProcessor.run()
    boundary : dict, optional  Output from CorrelationAnalyser.locking_boundary()
    """
    _set_style()
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(11, 7))

    # colour map: red = not locked, green = locked
    cmap  = mpl.cm.RdYlGn
    norm  = mcolors.Normalize(vmin=-3, vmax=3)

    # Marker styles by planet type
    markers = {"rocky": "o", "ice_giant": "s", "gas_giant": "^"}
    sizes   = {"rocky": 120, "ice_giant": 100, "gas_giant": 90}

    for ptype, marker in markers.items():
        sub = df[df["planet_type"] == ptype] if "planet_type" in df.columns else df
        if len(sub) == 0:
            continue
        log_rat = sub["log10_age_to_tau"].values.astype(float)
        colors  = cmap(norm(log_rat))
        sc = ax.scatter(
            sub["a_AU"], sub["R_planet_Rearth"],
            c=log_rat, cmap=cmap, norm=norm,
            s=sizes[ptype], marker=marker, edgecolors="black", linewidths=0.6,
            label=ptype.replace("_", " ").title(), zorder=4, alpha=0.9
        )

    # Labels for notable planets
    notable = {"HD 219134 b", "HD 75732 e", "HD 39091 c",
               "HD 217987 b", "HD 95735 b", "HD 115617 b"}
    if "planet_name" in df.columns:
        for _, row in df.iterrows():
            if row["planet_name"] in notable:
                ax.annotate(
                    row["planet_name"],
                    (row["a_AU"], row["R_planet_Rearth"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="black",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7)
                )

    # Empirical locking boundary
    if boundary and boundary.get("a_star_AU") is not None:
        ax.plot(boundary["a_star_AU"], boundary["R_range"],
                color="black", linewidth=2.5, linestyle="--",
                label="Empirical locking boundary", zorder=5)
        # Fill locked / not-locked regions
        ax.fill_betweenx(boundary["R_range"], 0, boundary["a_star_AU"],
                         alpha=0.06, color="green", label="Locked region")
        ax.fill_betweenx(boundary["R_range"], boundary["a_star_AU"], 10,
                         alpha=0.06, color="red",   label="Not-locked region")

    cb = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax
    )
    cb.set_label(r"$\log_{10}(\mathrm{Age}\,/\,\tau_{\rm sync})$", fontsize=16)
    cb.ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
    cb.ax.text(1.05, 0.02, "Locked →", transform=cb.ax.transAxes,
               fontsize=9, color="darkgreen", rotation=90)

    ax.set_xscale("log")
    ax.set_xlabel("Semi-major Axis $a$ (AU)", fontsize=18)
    ax.set_ylabel(r"Planet Radius $R_p\,(R_\oplus)$", fontsize=18)
    ax.set_title("HWO Target Tidal Locking Map\n"
                 r"(Green = Locked  |  Red = Not Locked  |  Dashed = Empirical Boundary)",
                 fontsize=16, fontweight="bold")
    ax.set_xlim(5e-3, 15)
    ax.set_ylim(0.3, 5.5)
    ax.legend(frameon=True, edgecolor="black", fontsize=12, loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 10. Empirical Power-Law Fit Panels ───────────────────────────────
def plot_empirical_fit(fit_a, fit_R, save_path=None):
    """
    Two-panel plot showing power-law fits of log10(age/tau) vs
    log10(a) and log10(R).  Includes 1-sigma confidence bands and
    the theoretical Gladman (1996) slope for comparison.
    """
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: vs distance ─────────────────────────────────────────
    ax = axes[0]
    if fit_a.get("log_a") is not None:
        log_a   = fit_a["log_a"]
        log_rat = fit_a["log_rat"]

        # colour by locked state
        colors = ["forestgreen" if v >= 0 else "firebrick" for v in log_rat]
        ax.scatter(log_a, log_rat, c=colors, s=60, edgecolors="black",
                   linewidths=0.5, zorder=4, alpha=0.85)

        # empirical fit
        fit_x = fit_a["fit_log_a"]
        fit_y = fit_a["fit_log_rat"]
        ax.plot(fit_x, fit_y, color="black", linewidth=2.5,
                label=rf"Empirical: $\alpha={fit_a['alpha']:.2f}\pm{fit_a['alpha_err']:.2f}$")

        # Theoretical Gladman slope (alpha = -6)
        fit_y_th = fit_a["log_A"] + (-6.0) * fit_x
        ax.plot(fit_x, fit_y_th, color="steelblue", linewidth=2.0,
                linestyle="--", label=r"Theory (Gladman 1996): $\alpha=-6$")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, label="Locking threshold")
        ax.set_xlabel(r"$\log_{10}(a\,/\,\mathrm{AU})$", fontsize=18)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Age}\,/\,\tau_{\rm sync})$", fontsize=18)
        ax.set_title("Locking Ratio vs. Orbital Distance", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.text(0.03, 0.97,
                f"$r_{{\\rm Pearson}}={fit_a['r_pearson']:.2f}$\n"
                f"$r_{{\\rm Spearman}}={fit_a['r_spearman']:.2f}$",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="black", alpha=0.85))

    # ── Panel 2: vs radius ───────────────────────────────────────────
    ax = axes[1]
    if fit_R.get("log_R") is not None:
        log_R   = fit_R["log_R"]
        log_rat = fit_R["log_rat"]

        colors = ["forestgreen" if v >= 0 else "firebrick" for v in log_rat]
        ax.scatter(log_R, log_rat, c=colors, s=60, edgecolors="black",
                   linewidths=0.5, zorder=4, alpha=0.85)

        fit_x = fit_R["fit_log_R"]
        fit_y = fit_R["fit_log_rat"]
        ax.plot(fit_x, fit_y, color="black", linewidth=2.5,
                label=rf"Empirical: $\beta={fit_R['beta']:.2f}\pm{fit_R['beta_err']:.2f}$")

        fit_y_th = fit_R["log_B"] + 3.0 * fit_x
        ax.plot(fit_x, fit_y_th, color="saddlebrown", linewidth=2.0,
                linestyle="--", label=r"Theory (Gladman 1996): $\beta=+3$")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, label="Locking threshold")
        ax.set_xlabel(r"$\log_{10}(R_p\,/\,R_\oplus)$", fontsize=18)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Age}\,/\,\tau_{\rm sync})$", fontsize=18)
        ax.set_title("Locking Ratio vs. Planet Radius", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.text(0.03, 0.97,
                f"$r_{{\\rm Pearson}}={fit_R['r_pearson']:.2f}$\n"
                f"$r_{{\\rm Spearman}}={fit_R['r_spearman']:.2f}$",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="black", alpha=0.85))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 11. Wind Signature Population ────────────────────────────────────
def plot_wind_signature_population(df, save_path=None):
    """
    Bar / scatter chart showing predicted atmospheric wind Δv for each
    planet, coloured by locked/not-locked state.

    This is a direct observational prediction: planets with Δv > 0
    are candidates for detecting non-synchronous rotation via
    ingress-egress transmission spectroscopy (CRIRES+/ELT).
    """
    _set_style()

    # Sort by a_AU for meaningful x ordering
    df_s = df.sort_values("a_AU")
    names = df_s["planet_name"].values
    dv    = df_s["predicted_delta_v_kms"].values.astype(float)
    locked= df_s["is_locked"].values if "is_locked" in df_s.columns else np.zeros(len(dv), bool)
    colors = ["forestgreen" if lk else "firebrick" for lk in locked]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: bar chart of Δv per planet ────────────────────────────
    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(x, dv, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel(r"Predicted $|\Delta v_{\rm atm}|$ (km/s)", fontsize=16)
    ax.set_xlabel("Planet", fontsize=16)
    ax.set_title("Predicted Atmospheric Wind Signature\n"
                 "(Green = Locked: Δv≈0  |  Red = Free rotator: Δv>0)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.6,
               label="ESPRESSO detection limit (~0.1 km/s)")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5,
               label="CRIRES+ detection limit (~0.5 km/s)")
    ax.legend(fontsize=10)

    # ── Right: Δv vs a (scatter) ─────────────────────────────────────
    ax = axes[1]
    for lk, clr, lbl in [(True, "forestgreen", "Tidally Locked (Δv=0)"),
                          (False, "firebrick",   "Free Rotator (Δv>0)")]:
        mask = locked == lk
        if np.any(mask):
            ax.scatter(df_s["a_AU"].values[mask], dv[mask],
                       c=clr, s=80, edgecolors="black", linewidths=0.6,
                       label=lbl, zorder=4, alpha=0.9)
    ax.axhline(0.1, color="gray",  linestyle="--", alpha=0.6)
    ax.axhline(0.5, color="black", linestyle=":",  alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Semi-major Axis $a$ (AU)", fontsize=16)
    ax.set_ylabel(r"Predicted $|\Delta v_{\rm atm}|$ (km/s)", fontsize=16)
    ax.set_title("Wind Signature vs. Orbital Distance\n"
                 "(Observational prediction for CRIRES+/ELT)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 12. Locked vs Unlocked Side-by-Side Comparison ───────────────────
def plot_locked_unlocked_comparison(comparison_dict, save_path=None):
    """
    4-panel comparison of locked vs unlocked synthetic planets.

    Panels (left=locked, right=unlocked):
      Top row:    RV anomaly time-series
      Bottom row: Transmission spectra (ingress vs egress)

    This is the CORE diagnostic figure of the project.
    """
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, key in enumerate(["locked", "unlocked"]):
        data   = comparison_dict[key]
        rv     = data["rv_data"]
        spec   = data["spec_data"]
        params = data["params"]
        is_lk  = key == "locked"
        lk_str = "TIDALLY LOCKED" if is_lk else "FREE ROTATOR"
        clr_in = "steelblue"   if is_lk else "firebrick"
        clr_eg = "steelblue"   if is_lk else "steelblue"

        # ── RV anomaly ───────────────────────────────────────────────
        ax = axes[0, col]
        t   = rv["time_hrs"]
        colors_map = {"out-of-transit": "gray", "ingress": "firebrick",
                      "in-transit": "black", "egress": "steelblue"}
        for lbl, clr in colors_map.items():
            mask = np.array([l == lbl for l in rv["labels"]])
            if np.any(mask):
                ax.errorbar(t[mask], rv["rv_km_s"][mask],
                            yerr=rv["rv_err_km_s"][mask],
                            fmt="o", color=clr, markersize=5,
                            alpha=0.8, zorder=3)
        ax.plot(t, rv["rv_true_km_s"], "k--", linewidth=1.5,
                alpha=0.7, label="True signal")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.4)

        dv_rm = rv["v_ingress"] - rv["v_egress"]
        ax.set_title(f"{lk_str}\n"
                     f"$\\Delta v_{{\\rm atm}}$ = {dv_rm:+.2f} km/s",
                     fontsize=13, fontweight="bold",
                     color="darkgreen" if is_lk else "firebrick")
        ax.set_xlabel("Time from Mid-Transit (hours)", fontsize=13)
        ax.set_ylabel("RV Anomaly (km/s)", fontsize=13)
        ax.legend(fontsize=9)

        # ── Transmission spectra ─────────────────────────────────────
        ax = axes[1, col]
        wl = spec["wavelength_nm"]
        ax.plot(wl, spec["flux_ingress"], color="firebrick", linewidth=1.8,
                label=f"Ingress: v={spec['v_ingress_km_s']:+.1f} km/s")
        ax.plot(wl, spec["flux_egress"],  color="steelblue", linewidth=1.8,
                label=f"Egress:  v={spec['v_egress_km_s']:+.1f} km/s")
        ax.axvline(spec["line_center_nm"], color="black", linestyle=":",
                   alpha=0.6, label="Rest lambda")
        ax.axvline(spec["line_center_nm"] + spec["dl_ingress_nm"],
                   color="firebrick", linestyle="--", alpha=0.5)
        ax.axvline(spec["line_center_nm"] + spec["dl_egress_nm"],
                   color="steelblue",  linestyle="--", alpha=0.5)

        overlap = "Lines OVERLAP (co-rotating atm)" if is_lk else "Lines SHIFTED (super-rotation)"
        ax.set_title(f"Transmission Spectra — {overlap}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=13)
        ax.set_ylabel("Normalized Flux", fontsize=13)
        ax.legend(fontsize=9)

    fig.suptitle("Tidally Locked vs Free Rotator: Key Observational Differences",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 13. Extended RM Corner Plot (4-param) ────────────────────────────
def plot_rm_corner_extended(samples, save_path=None):
    """
    Corner plot for the 4-parameter extended RM MCMC:
    [v_sin_i, lambda, b, delta_rv0].

    The key feature to look for is the b-lambda covariance:
    a higher impact parameter can mimic a larger spin-orbit angle
    in the RM signal shape.
    """
    try:
        import corner
    except ImportError:
        print("  corner not installed: pip install corner")
        return None
    _set_style()
    labels = [r"$v\,\sin i$ (km/s)", r"$\lambda$ (rad)",
              r"$b$", r"$\Delta\,rv_0$ (km/s)"]
    fig = corner.corner(
        samples, labels=labels,
        show_titles=True, title_kwargs={"fontsize": 10},
        color="black",
        hist_kwargs={"color": "black", "linewidth": 1.2},
        quantiles=[0.16, 0.5, 0.84],
        plot_contours=True,
        contour_kwargs={"colors": "black", "linewidths": 1.0},
    )
    fig.suptitle(
        "Extended RM MCMC Posteriors (4-param)\n"
        r"Note $b$-$\lambda$ covariance: higher $b$ can mimic larger spin-orbit angle",
        fontsize=11, fontweight="bold", y=1.02
    )
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 14. RV Anomaly Gallery (multi-panel for all batch planets) ────────
def plot_rv_anomaly_gallery(rv_dir, planet_names, save_path=None,
                             n_cols=5):
    """
    Multi-panel gallery showing the RV anomaly plot for every planet.
    Reads already-saved per-planet PNG files from rv_dir and assembles
    them into a single summary figure.

    Parameters
    ----------
    rv_dir       : str   Directory containing *_rv.png files
    planet_names : list  List of planet names to include
    n_cols       : int   Number of columns in the gallery grid
    """
    import math as _math
    from matplotlib.image import imread

    # Collect existing PNG paths in the order of planet_names
    pngs = []
    for pname in planet_names:
        fn = os.path.join(rv_dir, f"{pname.replace(' ','_')}_rv.png")
        if os.path.exists(fn):
            pngs.append((pname, fn))

    if not pngs:
        print("  No per-planet RV PNG files found in", rv_dir)
        return None

    n     = len(pngs)
    n_rows = _math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (pname, fpath) in enumerate(pngs):
        r, c = divmod(idx, n_cols)
        ax   = axes[r, c]
        img  = imread(fpath)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(pname, fontsize=7, fontweight="bold", pad=2)

    # Hide unused panels
    for idx in range(len(pngs), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("RV Anomaly Gallery — All HWO Batch Planets",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 15. Doppler Decomposition Population Plot ─────────────────────────
def plot_doppler_decomposition(df, save_path=None):
    """
    Stacked horizontal bar chart showing Dv_rot_excess and Dv_super for
    each planet, sorted by total Dv signal strength.

    This is the novel diagnostic figure: it separates the ROTATION signature
    from the ATMOSPHERIC JET signature, making clear which planets are spin-
    state detectable via rotation vs via jet.

    Color coding:
      Blue  = Dv_rot_excess  (axial rotation contribution)
      Orange = Dv_super      (super-rotation jet contribution)
    Locked planets are shown with hatching (rotation contribution ~ 0).
    """
    _set_style()
    required = {"planet_name", "dv_rot_excess_kms", "dv_super_kms",
                 "dv_total_kms", "is_locked"}
    if not required.issubset(df.columns):
        print("  plot_doppler_decomposition: missing required columns")
        return None

    df_plot = df[df["dv_total_kms"] > 0].copy()
    df_plot = df_plot.sort_values("dv_total_kms", ascending=True)

    n   = len(df_plot)
    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.38)))

    y_pos  = np.arange(n)
    labels = df_plot["planet_name"].tolist()
    dv_rot = df_plot["dv_rot_excess_kms"].values
    dv_sup = df_plot["dv_super_kms"].values
    locked = df_plot["is_locked"].values

    # Stacked bars
    hatch_rot = ["//" if lk else "" for lk in locked]
    bars1 = ax.barh(y_pos, dv_rot, color="steelblue", alpha=0.85,
                    label=r"$\Delta v_\mathrm{rot}$ (axial spin excess)")
    bars2 = ax.barh(y_pos, dv_sup, left=dv_rot, color="darkorange", alpha=0.85,
                    label=r"$\Delta v_\mathrm{super}$ (atmospheric jet)")

    # Hatch locked planets to distinguish
    for bar, lk in zip(bars1, locked):
        if lk:
            bar.set_hatch("///")
            bar.set_edgecolor("navy")

    # Vertical detection threshold lines
    for threshold, ls, label in [(0.1, "--", "0.1 km/s"),
                                  (1.0, ":",  "1.0 km/s (ELT limit)")]:
        ax.axvline(threshold, color="gray", linestyle=ls, linewidth=1.2,
                   alpha=0.7, label=f"Threshold {label}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r"Doppler Asymmetry $\Delta v$ (km/s)", fontsize=13)
    ax.set_title("Atmospheric Doppler Decomposition: Axial Rotation vs Super-Rotation\n"
                 "Hatched = tidally locked (rotation excess $\\approx 0$); "
                 "Solid = free rotator",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=0)

    # Annotate dominant effect for top 5
    top5 = df_plot.nlargest(5, "dv_total_kms")
    for _, row in top5.iterrows():
        idx = labels.index(row["planet_name"])
        dom = "ROT" if row["dv_rot_excess_kms"] > row["dv_super_kms"] else "JET"
        ax.text(row["dv_total_kms"] + 0.02, idx, dom,
                va="center", fontsize=7, color="black", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


# ── 16. Detection Feasibility Priority Map ────────────────────────────
def plot_detection_feasibility(df_feas, df_batch=None, save_path=None):
    """
    2-panel feasibility figure:

    Left panel: Scatter of Dv_total vs sigma_v per transit, colored by
    number of transits needed. The diagonal lines mark N_transit contours.
    This shows the detection landscape.

    Right panel: Ranked bar chart of top-20 most detectable planets,
    colored by whether they are locked or free (if df_batch provided).
    """
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: scatter detection landscape ────────────────────────────
    ax = axes[0]
    n_tr = df_feas["N_transits_needed"].values.astype(float)
    n_tr_clipped = np.clip(n_tr, 1, 100)

    sc = ax.scatter(
        df_feas["sigma_v_kms"], df_feas["dv_total_kms"],
        c=np.log10(n_tr_clipped), cmap="RdYlGn_r",
        s=60, alpha=0.85, edgecolors="black", linewidths=0.4, zorder=3
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}$(N transits needed)", fontsize=11)

    # N_transit contour lines: Dv = N_sigma * sigma_v / sqrt(N_tr)
    sv_range = np.logspace(np.log10(1e-5), np.log10(1e-2), 100)
    for N, ls in [(1, ":"), (5, "--"), (20, "-")]:
        dv_contour = 5.0 * sv_range / np.sqrt(N)
        ax.plot(sv_range, dv_contour, linestyle=ls, color="black",
                linewidth=1.0, alpha=0.6, label=f"N={N} transits")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_v$ per transit (km/s)", fontsize=12)
    ax.set_ylabel(r"$\Delta v_\mathrm{total}$ (km/s)", fontsize=12)
    ax.set_title("Detection Landscape: Signal vs Noise\n"
                 "(Green = easy, Red = many transits needed)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right: ranked bar chart of top 20 ────────────────────────────
    ax = axes[1]
    top20 = df_feas.head(20).copy()
    y_pos = np.arange(len(top20))

    # Color by locked status if df_batch provided
    if df_batch is not None and "is_locked" in df_batch.columns:
        merged = top20.merge(
            df_batch[["planet_name", "is_locked"]], on="planet_name", how="left"
        )
        colors = ["firebrick" if not lk else "steelblue"
                  for lk in merged["is_locked"].fillna(False)]
    else:
        colors = ["steelblue"] * len(top20)

    bars = ax.barh(y_pos, top20["dv_total_kms"].values,
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Annotate N_transits
    for bar, (_, row) in zip(bars, top20.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                f"N={int(row['N_transits_needed'])}",
                va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["planet_name"].tolist(), fontsize=8)
    ax.set_xlabel(r"$\Delta v_\mathrm{total}$ (km/s)", fontsize=12)
    ax.set_title("Top 20 Priority Targets for Spin-State Detection\n"
                 r"Red = free rotator, Blue = locked",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.6,
               label="0.1 km/s (ELT threshold)")
    ax.legend(fontsize=9)

    fig.suptitle(f"HWO Spin-State Detection Feasibility  "
                 f"({df_feas['instrument'].iloc[0] if len(df_feas) else 'HWO 6m'})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    return fig
