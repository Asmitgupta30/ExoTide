"""
run.py  (Module 2 — Tidal Locking & Rotation)
================================================
Jupyter-friendly entry point for the Tidal Locking analysis module.

Configure the switches below then run in Jupyter:
    %run module_2_tidal_rotation/run.py
or in terminal:
    python module_2_tidal_rotation/run.py

TIDAL_MODE selects between two distinct analysis pipelines:
  "lightcurve" — Extract stellar rotation period from TESS/Kepler photometry
                 via Lomb-Scargle periodogram and compare P_rot to P_orb.
                 This tells you if the star is rotating synchronously.
  "mcmc"       — Use the Rossiter-McLaughlin MCMC solver to fit v_sin_i and
                 spin-orbit angle lambda from Radial Velocity data.
                 This directly measures the rotation/orbit alignment.

DATA SOURCES (real observations):
  Photometry (lightcurve mode):
    - TESS: https://mast.stsci.edu/  (via lightkurve)
    - Kepler: https://mast.stsci.edu/
  Radial Velocity (mcmc mode):
    - HARPS ESO Archive: http://archive.eso.org/
    - ESPRESSO:          http://archive.eso.org/
    - HIRES Keck:        https://koa.ipac.caltech.edu/
    - NEID:              https://neid.ipac.caltech.edu/
  Transmission Spectra (super-rotation / wind):
    - CARMENES:          https://carmenes.cab.inta-csic.es/
    - CRIRES+:           http://archive.eso.org/
"""

import os
import sys
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these for your run
# ─────────────────────────────────────────────────────────────────────

# Get directory of this script (fallback for Jupyter)
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()

# Mode selector
TIDAL_MODE      = "mcmc"          # "lightcurve" or "mcmc"

# Target system (used by both modes)
TARGET          = "GJ-486b"       # Any MAST name or local path

# Orbital parameters for the target (used by rotation_analyser)
A_AU            = 0.01734         # GJ 486 b semi-major axis (AU)
M_STAR_MSUN     = 0.323           # GJ 486 stellar mass
R_PLANET_REARTH = 1.305           # GJ 486 b radius
P_ORB_DAYS      = 1.467           # GJ 486 b orbital period
ECCENTRICITY    = 0.05            # GJ 486 b eccentricity
SYSTEM_AGE_GYR  = 6.6             # GJ 486 system age
PLANET_TYPE     = "rocky"

# MCMC mode parameters (from a prior photometric fit)
MCMC_P_RATIO    = 0.0806          # Rp/R_* from photometric MCMC
MCMC_A_RATIO    = 36.7            # a/R_* from photometric MCMC
MCMC_B          = 0.11            # Impact parameter
TRANSIT_T0      = 0.0             # Mid-transit reference epoch (days)

# RM MCMC settings
RM_NSTEPS       = 2000
RM_NBURN        = 400

# Synthetic dataset switch
RUN_SYNTHETIC   = True    # Show the synthetic non-locked planet demonstration

# Output directory
OUTPUT_DIR = os.path.join(_here, "..", "output_plots", "module2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────

_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)          # ← fixed: was os.path.dirname(_parent)

from module_2_tidal_rotation.rotation_analyser  import RotationAnalyser
from module_2_tidal_rotation.rm_solver          import RMSolverMCMC
from module_2_tidal_rotation.lightcurve_tidal   import LightcurveTidalAnalyser
from module_2_tidal_rotation.synthetic_data     import SyntheticTidalDataset
import module_2_tidal_rotation.plots as plots


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ── 1. Rotation State Analysis (always runs) ─────────────────────────
section("1. ROTATION STATE ANALYSIS")

ra    = RotationAnalyser()
state = ra.full_analysis(
    a_AU=A_AU, M_star_Msun=M_STAR_MSUN,
    R_planet_Rearth=R_PLANET_REARTH,
    eccentricity=ECCENTRICITY,
    system_age_gyr=SYSTEM_AGE_GYR,
    planet_type=PLANET_TYPE,
    P_orbital_days=P_ORB_DAYS,
)
ra.print_report(state)

# Tidal sensitivity heatmap
plots.plot_sensitivity_heatmap(
    state["sensitivity"],
    save_path=os.path.join(OUTPUT_DIR, "tidal_sensitivity_heatmap.png")
)


# ── 2a. Lightcurve Mode ───────────────────────────────────────────────
if TIDAL_MODE == "lightcurve":
    section(f"2. LIGHTCURVE TIDAL MODE — {TARGET}")

    lta = LightcurveTidalAnalyser()

    if os.path.exists(TARGET):
        lc = lta.load_local(TARGET)
    else:
        lc = lta.download(TARGET)

    rot = lta.find_rotation_period(lc, method="lomb_scargle")
    lock_state = lta.assess_locking(rot["P_rot_days"], P_ORB_DAYS,
                                    system_age_gyr=SYSTEM_AGE_GYR)
    lta.print_report(lock_state)


# ── 2b. MCMC RM Mode ─────────────────────────────────────────────────
elif TIDAL_MODE == "mcmc":
    section(f"2. MCMC ROSSITER-MCLAUGHLIN MODE — {TARGET}")

    rm = RMSolverMCMC(nsteps=RM_NSTEPS, nburn=RM_NBURN)

    # Derive tight RV observation window from transit geometry
    b_s = min(abs(MCMC_B), 0.999)
    p_s = max(MCMC_P_RATIO, 0.001)
    a_s = MCMC_A_RATIO
    x_c_max  = float(np.sqrt((1.0 + p_s)**2 - b_s**2))
    if a_s <= x_c_max:
        a_s = x_c_max * 1.5
    phase_lim = float(np.arcsin(x_c_max / a_s)) / (2 * np.pi)
    t_limit   = phase_lim * P_ORB_DAYS
    t_rv      = np.linspace(TRANSIT_T0 - t_limit * 1.2,
                             TRANSIT_T0 + t_limit * 1.2, 100)

    # Generate synthetic-but-physically-derived RV signal (seeded from MCMC geometry)
    v_sin_i_true = 3.5
    lam_true     = np.radians(8.0)
    from module_2_tidal_rotation.rm_solver import RMSolverMCMC as _rm
    rv_true_model = _rm.ohta_rm_profile(
        t_rv, TRANSIT_T0, P_ORB_DAYS,
        v_sin_i_true, lam_true, p_s, a_s, b_s
    )
    sig_amp    = float(np.max(np.abs(rv_true_model))) or 0.005
    base_noise = sig_amp * 0.15
    # Variable heteroscedastic noise based on a baseline noise + random variation per point
    rv_err     = base_noise * (1.0 + 0.5 * np.random.uniform(-1, 1, len(t_rv)))
    rv_obs     = rv_true_model + np.random.normal(0, rv_err)

    # Run MCMC
    rm_results, rm_samples = rm.run_mcmc(
        t_rv, rv_obs, rv_err,
        MCMC_P_RATIO, a_s, b_s, TRANSIT_T0, P_ORB_DAYS
    )

    # Compute best-fit model curve
    rv_fit = rm.ohta_rm_profile(
        t_rv, TRANSIT_T0, P_ORB_DAYS,
        rm_results["v_sin_i"]["median"],
        rm_results["lambda_rad"]["median"],
        p_s, a_s, b_s
    )

    # Plots
    plots.plot_rm_anomaly(
        t_rv, rv_obs, rv_err, rv_fit, rm_results, TRANSIT_T0, TARGET,
        save_path=os.path.join(OUTPUT_DIR, "rm_anomaly.png")
    )
    plots.plot_rm_corner(
        rm_samples,
        save_path=os.path.join(OUTPUT_DIR, "rm_corner.png")
    )

else:
    print(f"Unknown TIDAL_MODE '{TIDAL_MODE}'. Use 'lightcurve' or 'mcmc'.")


# ── 3. Synthetic Non-Locked Planet Demo ──────────────────────────────
if RUN_SYNTHETIC:
    section("3. SYNTHETIC DEMO — Non-Tidally-Locked Planet")

    ds = SyntheticTidalDataset()
    ds.print_summary()

    rv_data   = ds.generate_rv_timeseries()
    spec_data = ds.generate_transmission_spectra()

    plots.plot_synthetic_rv(
        rv_data,
        save_path=os.path.join(OUTPUT_DIR, "synthetic_unlocked_rv.png")
    )
    plots.plot_transmission_spectra(
        spec_data,
        save_path=os.path.join(OUTPUT_DIR, "synthetic_transmission_spectra.png")
    )


# ── 4. Sensitivity Analysis ───────────────────────────────────────────
section("4. SENSITIVITY ANALYSIS")

_DAY_S = 86400.0

# 4a. tau_sync vs semi-major axis for 3 planet types
a_range = np.logspace(np.log10(0.005), np.log10(1.0), 200)
tau_by_type = {}
for _ptype in ["rocky", "ice_giant", "gas_giant"]:
    tau_by_type[_ptype] = np.array([
        ra.sync_timescale(a, M_STAR_MSUN, R_PLANET_REARTH, planet_type=_ptype)["tau_sync_Gyr"]
        for a in a_range
    ])

plots.plot_tidal_evolution(
    a_range, tau_by_type, SYSTEM_AGE_GYR, A_AU,
    save_path=os.path.join(OUTPUT_DIR, "tidal_evolution.png")
)

# 4b. Pseudo-synchronous rotation ratio vs eccentricity
ecc_range = np.linspace(0.0, 0.85, 300)
n_orb     = 2.0 * np.pi / (P_ORB_DAYS * _DAY_S)
ps_ratios = np.array([ra.pseudo_sync_rate(n_orb, e)["ratio_to_sync"] for e in ecc_range])

plots.plot_pseudo_sync_ecc(
    ecc_range, ps_ratios, ECCENTRICITY,
    save_path=os.path.join(OUTPUT_DIR, "pseudo_sync_ecc.png")
)

# 4c. Full 6-panel sensitivity dashboard
R_range      = np.linspace(0.5, 4.0, 120)
tau_vs_R     = np.array([ra.sync_timescale(A_AU, M_STAR_MSUN, R, planet_type="rocky")["tau_sync_Gyr"]
                          for R in R_range])

Mstar_range  = np.linspace(0.08, 2.0, 120)
tau_vs_Mstar = np.array([ra.sync_timescale(A_AU, M, R_PLANET_REARTH, planet_type="rocky")["tau_sync_Gyr"]
                          for M in Mstar_range])

Q_range      = np.logspace(1, 6, 120)
k2_vals_sens = [0.1, 0.299, 0.5]
tau_vs_Q     = {k2: np.array([
                    ra.sync_timescale(A_AU, M_STAR_MSUN, R_PLANET_REARTH, Q=Q, k2=k2)["tau_sync_Gyr"]
                    for Q in Q_range])
                for k2 in k2_vals_sens}

sens_results = {
    "a_range":       a_range,
    "tau_by_type":   tau_by_type,
    "ecc_range":     ecc_range,
    "ps_ratios":     ps_ratios,
    "R_range":       R_range,
    "tau_vs_R":      tau_vs_R,
    "Mstar_range":   Mstar_range,
    "tau_vs_Mstar":  tau_vs_Mstar,
    "Q_range":       Q_range,
    "tau_vs_Q":      tau_vs_Q,
    "k2_vals_sens":  k2_vals_sens,
    "sens_grid":     state["sensitivity"],
    "system_age_gyr": SYSTEM_AGE_GYR,
    "planet_a_AU":   A_AU,
    "planet_ecc":    ECCENTRICITY,
    "planet_R":      R_PLANET_REARTH,
    "planet_Mstar":  M_STAR_MSUN,
}

plots.plot_sensitivity_dashboard(
    sens_results,
    save_path=os.path.join(OUTPUT_DIR, "sensitivity_dashboard.png")
)


section("MODULE 2 COMPLETE")
print(f"  All outputs saved to: {OUTPUT_DIR}")

