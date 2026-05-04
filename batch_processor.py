"""
batch_processor.py  (Module 2 — Tidal Locking)
================================================
Reads hwo_targets.xlsx and runs the full tidal locking analysis
for every planet row, saving results to batch_results.csv.

Usage
-----
from module_2_tidal_rotation.batch_processor import BatchTidalProcessor
bp = BatchTidalProcessor("module_2_tidal_rotation/hwo_targets.xlsx")
df = bp.run()
bp.save(df, "output_plots/module2/batch/batch_results.csv")

References for system ages used in create_hwo_sheet.py
-------------------------------------------------------
- HD 10700 (tau Ceti):   Tuomi et al. 2013, A&A — 5.8 Gyr
- HD 75732 (55 Cnc):     Bourrier et al. 2018 — 10.2 Gyr
- HD 219134:             Motalebi et al. 2015 — 10.8 Gyr
- HD 143761 (rho CrB):   Brewer et al. 2023 — 11.5 Gyr
- HD 22049 (eps Eri):    Mamajek & Hillenbrand 2008 — 0.7 Gyr
- All others: estimated from stellar Teff / literature
"""

import os
import math
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from module_2_tidal_rotation.rotation_analyser import RotationAnalyser
from module_2_tidal_rotation.rm_solver import (
    RMSolverMCMC, RMSolverMCMCExtended,
    estimate_v_sin_i, estimate_geometric_params
)

# Physical constants
G      = 6.674e-11
M_SUN  = 1.989e30
AU     = 1.496e11
DAY_S  = 86400.0
MJ_ME  = 317.83        # Jupiter mass -> Earth mass
C_KMS  = 2.998e5       # km/s



class BatchTidalProcessor:
    """
    Runs tidal locking analysis for every planet in an Excel/CSV sheet.

    Parameters
    ----------
    excel_path : str   Path to hwo_targets.xlsx (or .csv)
    """

    def __init__(self, excel_path: str):
        if not HAS_PANDAS:
            raise ImportError("pandas is required for batch processing: pip install pandas openpyxl")
        self.excel_path = excel_path
        self.ra = RotationAnalyser()

    # ── Data loader ───────────────────────────────────────────────────────
    def load_sheet(self) -> "pd.DataFrame":
        path = self.excel_path
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name="HWO Targets")

        # Validate required columns
        required = ["planet_name", "M_star_Msun", "M_planet_Mearth",
                    "a_AU", "eccentricity", "system_age_gyr", "planet_type"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in sheet: {missing}")
        return df

    # ── Kepler period ─────────────────────────────────────────────────────
    @staticmethod
    def kepler_period_days(a_AU: float, M_star_Msun: float) -> float:
        a_m = a_AU * AU
        M_s = M_star_Msun * M_SUN
        return (2 * math.pi * math.sqrt(a_m**3 / (G * M_s))) / DAY_S

    # ── Forward-model atmospheric wind Δv ────────────────────────────────
    @staticmethod
    def predict_wind_delta_v(a_AU: float, M_star_Msun: float,
                             R_planet_Rearth: float, is_locked: bool) -> float:
        """
        Estimate the ingress-egress Doppler shift (km/s) a planet would
        show IF it had an active super-rotating atmosphere.

        For a tidally locked planet: Δv ≈ 0 (symmetric atmosphere)
        For a free-rotator: Δv scales with the atmospheric wind speed,
        which is approximated as a fraction of the orbital velocity.

        The scaling follows Showman & Guillot (2002): wind speed ∝ (R_p / a)^0.5
        normalised to the synthetic dataset fiducial (HD 209458b-like hot Jupiter).

        Returns
        -------
        float   Predicted |Δv_atm| in km/s  (0 if locked)
        """
        if is_locked:
            return 0.0
        # Orbital velocity of the planet
        a_m  = a_AU * AU
        M_s  = M_star_Msun * M_SUN
        v_orb_kms = math.sqrt(G * M_s / a_m) / 1e3  # km/s

        # Wind speed as fraction of orbital velocity, scaled by size
        # Fiducial: HD 209458b → Δv~2 km/s, v_orb~152 km/s, R~1.35 R_J~15 R_E
        R_fiducial = 15.0   # R_Earth
        v_orb_fid  = 152.0  # km/s
        dv_fid     = 4.0    # km/s (ingress-egress separation)

        dv = dv_fid * (v_orb_kms / v_orb_fid) * math.sqrt(R_planet_Rearth / R_fiducial)
        return float(dv)

    # ── Single-planet analysis ────────────────────────────────────────────
    def analyse_one(self, row: dict) -> dict:
        a       = float(row["a_AU"])
        Mstar   = float(row["M_star_Msun"])
        ME      = float(row["M_planet_Mearth"])
        ecc     = float(row["eccentricity"])
        age     = float(row["system_age_gyr"])
        ptype   = str(row["planet_type"])
        pname   = str(row["planet_name"])

        # Derive radius from Chen & Kipping (already in RotationAnalyser)
        # We call sync_timescale with M_planet_Mearth so it skips the M-R law
        # for mass, but we still need radius — derive it here
        if ME < 1.23**( 1/0.55) * 1.008**(1/0.55):
            R_earth = 1.008 * (ME)**(0.55)
        elif ME < (14.26 / 0.808)**(1/0.589):
            R_earth = 0.808 * (ME)**(0.589)
        else:
            R_earth = max(1.0, ME * 0.02)   # rough fallback

        P_days = self.kepler_period_days(a, Mstar)

        state = self.ra.full_analysis(
            a_AU=a,
            M_star_Msun=Mstar,
            R_planet_Rearth=R_earth,
            eccentricity=ecc,
            system_age_gyr=age,
            planet_type=ptype,
            P_orbital_days=P_days,
            M_planet_Mearth=ME,
        )

        is_locked      = state["is_locked"]
        age_to_tau     = state["age_to_tau"]
        locked_frac    = state["sensitivity"]["locked_fraction"]
        tau_gyr        = state["sync"]["tau_sync_Gyr"]
        ps_ratio       = state["pseudo_sync"]["ratio_to_sync"]
        ps_period      = state["pseudo_sync"]["P_rotation_days"]
        label          = state["state_label"]
        resonances     = "; ".join(r["resonance"] for r in state["resonances"]) or "none"

        # ── Error propagation (measurement uncertainties only) ──────────────
        tau_err = self.ra.sync_timescale_with_errors(
            a, Mstar, R_earth, eccentricity=ecc,
            planet_type=ptype, M_planet_Mearth=ME
        )
        tau_lo  = tau_err["tau_sync_Gyr_lo"]
        tau_hi  = tau_err["tau_sync_Gyr_hi"]
        frac_e  = tau_err["frac_err_meas"]

        # age/tau bounds from measurement errors
        rat_lo  = age / tau_hi if tau_hi > 0 else 0.0
        rat_hi  = age / tau_lo if tau_lo > 0 else float("inf")
        log_lo  = math.log10(max(rat_lo, 1e-9))
        log_hi  = math.log10(max(rat_hi, 1e-9))

        # Q-model bounds: Q spans 1 dex -> tau spans 1 dex -> ratio spans 1 dex
        log_rat = math.log10(max(age_to_tau, 1e-9))
        log_Q_lo = log_rat - 1.0   # lower age/tau (larger tau, larger Q)
        log_Q_hi = log_rat + 1.0   # upper age/tau (smaller tau, smaller Q)

        dv = self.predict_wind_delta_v(a, Mstar, R_earth, is_locked)

        # Doppler decomposition: axial rotation vs super-rotation jet
        P_rot_days = state["pseudo_sync"]["P_rotation_days"]
        Teff       = float(row.get("Teff_K", 5500))
        dcomp = self.ra.decompose_doppler_signal(
            a, Mstar, R_earth,
            P_rot_days=P_rot_days,
            P_orb_days=P_days,
            Teff_K=Teff,
            is_locked=is_locked,
        )

        return {
            "planet_name":             pname,
            "star_name":               row.get("star_name", ""),
            "star_common":             row.get("star_common", ""),
            "M_star_Msun":             Mstar,
            "Teff_K":                  row.get("Teff_K", float("nan")),
            "M_planet_Mearth":         round(ME, 3),
            "R_planet_Rearth":         round(R_earth, 3),
            "a_AU":                    a,
            "eccentricity":            ecc,
            "system_age_gyr":          age,
            "P_orb_days":              round(P_days, 4),
            "planet_type":             ptype,
            "tau_sync_Gyr":            round(tau_gyr, 6),
            "tau_sync_Gyr_lo":         round(tau_lo, 6),
            "tau_sync_Gyr_hi":         round(tau_hi, 6),
            "frac_err_meas":           round(frac_e, 3),
            "age_to_tau":              round(age_to_tau, 4),
            "age_to_tau_lo":           round(rat_lo, 4),
            "age_to_tau_hi":           round(rat_hi, 4),
            "log10_age_to_tau":        round(log_rat, 4),
            "log10_age_to_tau_meas_lo": round(log_lo, 4),
            "log10_age_to_tau_meas_hi": round(log_hi, 4),
            "log10_age_to_tau_Q_lo":    round(log_Q_lo, 4),
            "log10_age_to_tau_Q_hi":    round(log_Q_hi, 4),
            "is_locked":               is_locked,
            "locked_fraction_Qk2":     round(locked_frac, 3),
            "ps_ratio":                round(ps_ratio, 4),
            "ps_period_days":          round(ps_period, 4),
            "state_label":             label,
            "resonances":              resonances,
            "predicted_delta_v_kms":   round(dv, 3),
            # Doppler decomposition
            "v_eq_kms":                dcomp["v_eq_kms"],
            "v_eq_locked_kms":         dcomp["v_eq_locked_kms"],
            "dv_rot_excess_kms":       dcomp["dv_rot_excess_kms"],
            "dv_super_kms":            dcomp["dv_super_kms"],
            "dv_total_kms":            dcomp["dv_total_kms"],
            "T_eq_K":                  dcomp["T_eq_K"],
            "rot_dominates":           dcomp["rot_dominates"],
            "notes":                   row.get("notes", ""),
        }

    # ── Batch run ─────────────────────────────────────────────────────────
    def run(self, planet_types=None, verbose=True) -> "pd.DataFrame":
        """
        Run analysis for all (or filtered) rows.

        Parameters
        ----------
        planet_types : list of str, optional
            Filter to specific types, e.g. ['rocky']. None = all.
        verbose : bool

        Returns
        -------
        pd.DataFrame   One row per planet with all derived quantities.
        """
        df_in = self.load_sheet()
        if planet_types:
            df_in = df_in[df_in["planet_type"].isin(planet_types)]

        results = []
        n = len(df_in)
        for i, (_, row) in enumerate(df_in.iterrows()):
            if verbose:
                print(f"  [{i+1:3d}/{n}] {row['planet_name']} ...", end=" ")
            try:
                res = self.analyse_one(row.to_dict())
                results.append(res)
                if verbose:
                    state = "LOCKED" if res["is_locked"] else "free  "
                    print(f"{state}  (age/tau = {res['age_to_tau']:.3g})")
            except Exception as exc:
                if verbose:
                    print(f"ERROR: {exc}")
                results.append({"planet_name": row.get("planet_name", "?"),
                                "error": str(exc)})

        return pd.DataFrame(results)

    # ── Save ──────────────────────────────────────────────────────────────
    @staticmethod
    def save(df: "pd.DataFrame", csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Batch results saved: {csv_path}")
        locked = df["is_locked"].sum() if "is_locked" in df.columns else "?"
        print(f"  Locked: {locked}/{len(df)} planets")

    # ── Batch RM MCMC ─────────────────────────────────────────────────────
    def run_rm_batch(self, df_batch: "pd.DataFrame",
                     out_dir: str,
                     nsteps: int = 500,
                     nburn:  int = 400,
                     extended: bool = False,
                     verbose: bool = True) -> "pd.DataFrame":
        """
        For every planet in df_batch, generate synthetic RV data and
        run a fast RM MCMC.  Saves per-planet RV anomaly plots.

        Parameters
        ----------
        df_batch  : pd.DataFrame  Output from run()
        out_dir   : str           Directory for per-planet PNG files
        nsteps    : int           MCMC steps (default 500 — fast batch)
        nburn     : int           Burn-in steps
        extended  : bool          If True, use 4-param RMSolverMCMCExtended
        verbose   : bool

        Returns
        -------
        pd.DataFrame  Input df_batch with added RM result columns
        """
        import module_2_tidal_rotation.plots as plots

        rv_dir  = os.path.join(out_dir, "rv_anomaly")
        os.makedirs(rv_dir, exist_ok=True)

        # Force non-interactive backend so savefig() works in notebooks/batch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import module_2_tidal_rotation.plots as plots

        # MCMC defaults: extended=True -> rigorous 10k steps / 64 walkers
        #               extended=False -> fast 2k steps / 32 walkers
        if extended:
            _nwalkers = 64
            _nsteps   = nsteps   # caller sets 10000
            _nburn    = max(nburn, int(_nsteps * 0.3))
            _nthin    = 15
        else:
            _nwalkers = 32
            _nsteps   = nsteps
            _nburn    = nburn
            _nthin    = 10

        n = len(df_batch)
        rm_records = []

        for i, (_, row) in enumerate(df_batch.iterrows()):
            pname = str(row.get("planet_name", f"planet_{i}"))
            if verbose:
                print(f"  RM [{i+1:3d}/{n}] {pname} ...", end=" ", flush=True)

            rec = {"planet_name": pname}

            try:
                a      = float(row["a_AU"])
                Mstar  = float(row["M_star_Msun"])
                R_p    = float(row["R_planet_Rearth"])
                P_days = float(row["P_orb_days"])
                Teff   = float(row.get("Teff_K", 5500))

                # Geometric params from stellar mass + planet size
                geom      = estimate_geometric_params(R_p, Mstar, a)
                p_r       = geom["p_ratio"]
                a_r       = geom["a_ratio"]
                b_init    = geom["b"]
                v_si_true = estimate_v_sin_i(Teff)

                # Clip a_ratio so arcsin is defined for transit window
                x_c_max   = float(np.sqrt(max((1.0 + p_r)**2 - b_init**2, 1e-6)))
                a_r       = max(a_r, x_c_max * 1.05)
                sin_arg   = min(x_c_max / a_r, 0.9999)
                phase_lim = float(np.arcsin(sin_arg)) / (2 * math.pi)
                t_lim     = phase_lim * P_days

                # Use 120 points in-window + 30 each side OOT for better coverage
                t_oot  = np.linspace(-t_lim * 2.0, -t_lim * 1.05, 30)
                t_in   = np.linspace(-t_lim, t_lim, 120)
                t_oot2 = np.linspace(t_lim * 1.05, t_lim * 2.0, 30)
                t_rv   = np.concatenate([t_oot, t_in, t_oot2])
                t0     = 0.0

                # Draw a small random spin-orbit angle (HWO targets mostly aligned)
                lam_true = np.radians(np.random.uniform(-20, 20))
                rv_true  = RMSolverMCMC.ohta_rm_profile(
                    t_rv, t0, P_days, v_si_true, lam_true, p_r, a_r, b_init
                )

                # 8% noise -> SNR ~12 at peak (much better than 18%)
                sig_amp    = max(float(np.max(np.abs(rv_true))), 0.003)
                base_noise = sig_amp * 0.08
                # Heteroscedastic: slightly larger errors at ingress/egress edges
                t_abs  = np.abs(t_rv)
                noise_scale = 1.0 + 0.4 * np.exp(-((t_abs - t_lim)**2) / (0.1*t_lim)**2)
                rv_err = base_noise * noise_scale
                rv_obs = rv_true + np.random.normal(0, rv_err)

                # Run MCMC
                if extended:
                    solver = RMSolverMCMCExtended(
                        nwalkers=_nwalkers, nsteps=_nsteps,
                        nburn=_nburn, nthin=_nthin
                    )
                    rm_res, rm_flat = solver.run_mcmc_extended(
                        t_rv, rv_obs, rv_err, p_r, a_r, b_init, t0, P_days
                    )
                    rec["b_med"]    = round(rm_res["b"]["median"], 3)
                    rec["b_err"]    = round(rm_res["b"]["upper_err"], 3)
                    rec["drv0_med"] = round(rm_res["delta_rv0"]["median"], 4)
                else:
                    solver = RMSolverMCMC(
                        nwalkers=_nwalkers, nsteps=_nsteps,
                        nburn=_nburn, nthin=_nthin
                    )
                    rm_res, rm_flat = solver.run_mcmc(
                        t_rv, rv_obs, rv_err, p_r, a_r, b_init, t0, P_days
                    )

                # Best-fit model for plotting
                rv_fit = RMSolverMCMC.ohta_rm_profile(
                    t_rv, t0, P_days,
                    rm_res["v_sin_i"]["median"],
                    rm_res["lambda_rad"]["median"],
                    p_r, a_r,
                    rm_res.get("b", {}).get("median", b_init)
                )

                # Per-planet RV anomaly plot — save then close immediately
                safe   = pname.replace(" ", "_").replace("/", "-")
                png_rv = os.path.join(rv_dir, f"{safe}_rv.png")
                fig_rv = plots.plot_rm_anomaly(
                    t_rv, rv_obs, rv_err, rv_fit, rm_res,
                    t0, pname, save_path=png_rv
                )
                if fig_rv is not None:
                    _plt.close(fig_rv)

                # RM corner plot
                png_corner = os.path.join(rv_dir, f"{safe}_corner.png")
                if extended:
                    fig_c = plots.plot_rm_corner_extended(
                        rm_flat, save_path=png_corner
                    )
                else:
                    fig_c = plots.plot_rm_corner(rm_flat, save_path=png_corner)
                if fig_c is not None:
                    _plt.close(fig_c)

                rec["v_sin_i_med"]      = round(rm_res["v_sin_i"]["median"], 3)
                rec["v_sin_i_err"]      = round(rm_res["v_sin_i"]["upper_err"], 3)
                rec["lambda_deg_med"]   = round(rm_res["lambda_deg"]["median"], 2)
                rec["lambda_deg_err"]   = round(rm_res["lambda_deg"]["upper_err"], 2)
                rec["lambda_true_deg"]  = round(float(np.degrees(lam_true)), 2)
                rec["rm_tidal_state"]   = rm_res.get("tidal_state", "")
                rec["rm_amplitude_kms"] = round(
                    rm_res.get("theoretical_amplitude_km_s", 0.0), 4)

                if verbose:
                    lam_r = rec["lambda_deg_med"]
                    print(f"v_si={rec['v_sin_i_med']:.2f} "
                          f"lam={lam_r:+.1f}d (true={rec['lambda_true_deg']:+.1f}d) "
                          f"-> {png_rv}")

            except Exception as exc:
                import traceback
                if verbose:
                    print(f"ERROR: {exc}")
                    traceback.print_exc()
                rec["rm_error"] = str(exc)

            rm_records.append(rec)

        rm_df  = pd.DataFrame(rm_records)
        merged = df_batch.merge(rm_df, on="planet_name", how="left")
        return merged

