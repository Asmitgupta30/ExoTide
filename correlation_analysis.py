"""
correlation_analysis.py  (Module 2 — Tidal Locking)
=====================================================
Reads batch_results.csv and performs:

  1. Power-law fits:  log(age/tau_sync) vs log(a) and log(R_p)
  2. 2-D surface fit in (a, R) space
  3. Locking boundary derivation (the a* where age/tau = 1 for each R)
  4. Wind signature prediction: predicted_delta_v vs a and R
  5. LaTeX table export

Usage
-----
from module_2_tidal_rotation.correlation_analysis import CorrelationAnalyser
ca = CorrelationAnalyser("output_plots/module2/batch/batch_results.csv")
results = ca.run_all(plot_dir="output_plots/module2/batch/")
ca.print_latex_table(results["df"])
"""

import os
import math
import warnings
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class CorrelationAnalyser:
    """
    Empirical correlation analysis on batch tidal locking results.

    The key outputs are:
      - Power-law exponents for tau_sync ~ a^alpha * R^beta
        (theory predicts alpha ~ 6, beta ~ -3 from Gladman 1996)
      - The empirical locking boundary in (a, R) space
      - Correlation of predicted atmospheric wind Δv with locking state
    """

    def __init__(self, csv_path: str):
        if not HAS_PANDAS:
            raise ImportError("pandas required: pip install pandas")
        self.csv_path = csv_path
        self.df_raw   = None

    # ── Load ──────────────────────────────────────────────────────────────
    def load(self) -> "pd.DataFrame":
        df = pd.read_csv(self.csv_path)
        # Drop rows with errors or missing critical columns
        df = df[df["tau_sync_Gyr"].notna() & df["a_AU"].notna() &
                df["R_planet_Rearth"].notna()]
        df = df[df["tau_sync_Gyr"] > 0]
        self.df_raw = df
        return df

    # ── Power-law fit helpers ─────────────────────────────────────────────
    @staticmethod
    def _power_law_1d(log_x, log_A, alpha):
        """log(y) = log_A + alpha * log(x)"""
        return log_A + alpha * log_x

    @staticmethod
    def _power_law_2d(X, log_A, alpha, beta):
        """log(y) = log_A + alpha*log(a) + beta*log(R)"""
        log_a, log_R = X
        return log_A + alpha * log_a + beta * log_R

    # ── Fit: log(age/tau) vs log(a) ──────────────────────────────────────
    def fit_vs_distance(self, df: "pd.DataFrame") -> dict:
        """
        Fit log10(age/tau_sync) = A + alpha * log10(a_AU)

        Theory (Gladman 1996): tau_sync ∝ a^6  ->  age/tau ∝ a^{-6}
        So we expect alpha ~ -6.
        """
        mask = df["log10_age_to_tau"].notna() & np.isfinite(df["log10_age_to_tau"])
        sub  = df[mask]
        log_a    = np.log10(sub["a_AU"].values.astype(float))
        log_rat  = sub["log10_age_to_tau"].values.astype(float)

        result = {"n_points": len(sub), "alpha": None, "alpha_err": None,
                  "log_A": None, "r_pearson": None, "r_spearman": None}

        if not HAS_SCIPY or len(sub) < 4:
            return result

        try:
            popt, pcov = curve_fit(self._power_law_1d, log_a, log_rat,
                                   p0=[-1.0, -6.0], maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            rp, _  = pearsonr(log_a, log_rat)
            rs, _  = spearmanr(log_a, log_rat)
            result.update({
                "log_A":      float(popt[0]),
                "alpha":      float(popt[1]),
                "alpha_err":  float(perr[1]),
                "log_A_err":  float(perr[0]),
                "r_pearson":  float(rp),
                "r_spearman": float(rs),
                "log_a":      log_a,
                "log_rat":    log_rat,
                "fit_log_a":  np.linspace(log_a.min(), log_a.max(), 200),
            })
            result["fit_log_rat"] = (result["log_A"]
                                     + result["alpha"] * result["fit_log_a"])
        except Exception as e:
            warnings.warn(f"fit_vs_distance failed: {e}")

        return result

    # ── Fit: log(age/tau) vs log(R) ──────────────────────────────────────
    def fit_vs_radius(self, df: "pd.DataFrame") -> dict:
        """
        Fit log10(age/tau_sync) = B + beta * log10(R_planet_Rearth)

        Theory: tau_sync ∝ R^{-3}  ->  age/tau ∝ R^{+3}  -> beta ~ +3.
        """
        mask = df["log10_age_to_tau"].notna() & np.isfinite(df["log10_age_to_tau"])
        sub  = df[mask]
        log_R   = np.log10(sub["R_planet_Rearth"].values.astype(float))
        log_rat = sub["log10_age_to_tau"].values.astype(float)

        result = {"n_points": len(sub), "beta": None, "beta_err": None,
                  "log_B": None, "r_pearson": None, "r_spearman": None}

        if not HAS_SCIPY or len(sub) < 4:
            return result

        try:
            popt, pcov = curve_fit(self._power_law_1d, log_R, log_rat,
                                   p0=[0.0, 3.0], maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            rp, _  = pearsonr(log_R, log_rat)
            rs, _  = spearmanr(log_R, log_rat)
            result.update({
                "log_B":      float(popt[0]),
                "beta":       float(popt[1]),
                "beta_err":   float(perr[1]),
                "log_B_err":  float(perr[0]),
                "r_pearson":  float(rp),
                "r_spearman": float(rs),
                "log_R":      log_R,
                "log_rat":    log_rat,
                "fit_log_R":  np.linspace(log_R.min(), log_R.max(), 200),
            })
            result["fit_log_rat"] = (result["log_B"]
                                     + result["beta"] * result["fit_log_R"])
        except Exception as e:
            warnings.warn(f"fit_vs_radius failed: {e}")

        return result

    # ── 2D surface fit ────────────────────────────────────────────────────
    def fit_2d_surface(self, df: "pd.DataFrame") -> dict:
        """
        Fit log10(age/tau) = log_A + alpha*log10(a) + beta*log10(R)

        This gives the joint power-law surface.  The locking boundary
        (age/tau = 1, i.e. log10 = 0) is then:
            log10(a*) = (-log_A - beta*log10(R)) / alpha
        """
        mask = (df["log10_age_to_tau"].notna()
                & np.isfinite(df["log10_age_to_tau"])
                & np.isfinite(np.log10(df["R_planet_Rearth"].values.astype(float))))
        sub  = df[mask]

        log_a   = np.log10(sub["a_AU"].values.astype(float))
        log_R   = np.log10(sub["R_planet_Rearth"].values.astype(float))
        log_rat = sub["log10_age_to_tau"].values.astype(float)

        result = {"n_points": len(sub), "alpha": None, "beta": None, "log_A": None}

        if not HAS_SCIPY or len(sub) < 6:
            return result

        try:
            popt, pcov = curve_fit(self._power_law_2d,
                                   (log_a, log_R), log_rat,
                                   p0=[-1.0, -6.0, 3.0], maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            result.update({
                "log_A":     float(popt[0]),
                "alpha":     float(popt[1]),
                "beta":      float(popt[2]),
                "log_A_err": float(perr[0]),
                "alpha_err": float(perr[1]),
                "beta_err":  float(perr[2]),
            })
        except Exception as e:
            warnings.warn(f"fit_2d_surface failed: {e}")

        return result

    # ── Locking boundary ──────────────────────────────────────────────────
    def locking_boundary(self, fit_2d: dict, R_range=None) -> dict:
        """
        From the 2D fit, derive a*(R) — the semi-major axis at which
        a planet of radius R transitions from locked to unlocked.

        log10(a*) = (-log_A - beta * log10(R)) / alpha
        """
        if fit_2d.get("alpha") is None:
            return {}

        if R_range is None:
            R_range = np.linspace(0.5, 4.0, 200)

        log_A  = fit_2d["log_A"]
        alpha  = fit_2d["alpha"]
        beta   = fit_2d["beta"]

        log_a_star = (-log_A - beta * np.log10(R_range)) / alpha
        a_star     = 10.0 ** log_a_star

        return {
            "R_range":   R_range,
            "a_star_AU": a_star,
            "fit_2d":    fit_2d,
        }

    # ── Wind Δv correlation ───────────────────────────────────────────────
    def wind_correlation(self, df: "pd.DataFrame") -> dict:
        """
        Correlate predicted_delta_v_kms with a_AU and R_planet_Rearth.
        Planets that are locked should have Δv ~ 0;
        free rotators have non-zero Δv.
        """
        sub = df[df["predicted_delta_v_kms"].notna()]
        result = {
            "df_sub":          sub,
            "r_dv_vs_a":       None,
            "r_dv_vs_R":       None,
            "locked_mean_dv":  sub[sub["is_locked"]==True]["predicted_delta_v_kms"].mean(),
            "free_mean_dv":    sub[sub["is_locked"]==False]["predicted_delta_v_kms"].mean(),
        }
        if HAS_SCIPY and len(sub) > 3:
            try:
                r1, _ = pearsonr(sub["a_AU"], sub["predicted_delta_v_kms"])
                r2, _ = pearsonr(sub["R_planet_Rearth"],
                                 sub["predicted_delta_v_kms"])
                result["r_dv_vs_a"] = float(r1)
                result["r_dv_vs_R"] = float(r2)
            except Exception:
                pass
        return result

    # ── LaTeX table ───────────────────────────────────────────────────────
    @staticmethod
    def print_latex_table(df: "pd.DataFrame", rocky_only: bool = True):
        """
        Print a LaTeX table of tidal locking results suitable for insertion
        into a paper.
        """
        sub = df[df["planet_type"] == "rocky"] if rocky_only else df
        sub = sub.sort_values("a_AU")

        print("\n% ── Tidal Locking Results (LaTeX) ──────────────────")
        print("\\begin{table}[ht]")
        print("\\centering")
        print("\\caption{Tidal locking analysis for rocky HWO targets.}")
        print("\\label{tab:tidal_locking}")
        print("\\begin{tabular}{lrrrrcrc}")
        print("\\hline\\hline")
        print("Planet & $a$ (AU) & $R_p$ ($R_\\oplus$) & "
              "$\\tau_{\\rm sync}$ (Gyr) & Age/\\,$\\tau$ & Locked? & "
              "$f_{Q,k_2}$ & $\\Delta v$ (km/s) \\\\")
        print("\\hline")
        for _, row in sub.iterrows():
            locked_str = "\\checkmark" if row.get("is_locked") else "---"
            frac = row.get("locked_fraction_Qk2", float("nan"))
            frac_str = f"{frac:.0%}" if not math.isnan(frac) else "---"
            dv   = row.get("predicted_delta_v_kms", 0.0)
            dv_str = f"{dv:.2f}" if dv > 0.001 else "0"
            tau  = row.get("tau_sync_Gyr", float("nan"))
            rat  = row.get("age_to_tau", float("nan"))
            print(f"{row['planet_name']:20s} & "
                  f"{row['a_AU']:.4f} & "
                  f"{row['R_planet_Rearth']:.2f} & "
                  f"{tau:.2g} & "
                  f"{rat:.2g} & "
                  f"{locked_str:12s} & "
                  f"{frac_str} & "
                  f"{dv_str} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    # ── Print summary ─────────────────────────────────────────────────────
    @staticmethod
    def print_summary(fit_a: dict, fit_R: dict, fit_2d: dict, boundary: dict):
        print("\n" + "="*65)
        print("  EMPIRICAL CORRELATION RESULTS")
        print("="*65)

        if fit_a.get("alpha") is not None:
            print(f"\n  log10(age/tau) vs log10(a):")
            print(f"    alpha = {fit_a['alpha']:.2f} +- {fit_a['alpha_err']:.2f}")
            print(f"    Theory (Gladman 1996): alpha ~ -6")
            print(f"    Pearson r = {fit_a['r_pearson']:.3f}, "
                  f"Spearman r = {fit_a['r_spearman']:.3f}")

        if fit_R.get("beta") is not None:
            print(f"\n  log10(age/tau) vs log10(R):")
            print(f"    beta  = {fit_R['beta']:.2f} +- {fit_R['beta_err']:.2f}")
            print(f"    Theory (Gladman 1996): beta ~ +3")
            print(f"    Pearson r = {fit_R['r_pearson']:.3f}, "
                  f"Spearman r = {fit_R['r_spearman']:.3f}")

        if fit_2d.get("alpha") is not None:
            print(f"\n  2D surface fit:  log10(age/tau) = {fit_2d['log_A']:.2f}"
                  f" + {fit_2d['alpha']:.2f}·log10(a) + {fit_2d['beta']:.2f}·log10(R)")
            print(f"    alpha_err={fit_2d['alpha_err']:.2f}, beta_err={fit_2d['beta_err']:.2f}")

        print("="*65)

    # ── Master runner ─────────────────────────────────────────────────────
    def run_all(self, plot_dir: str = None, rocky_only: bool = False) -> dict:
        """
        Load data, run all fits, optionally produce plots.

        Parameters
        ----------
        plot_dir   : str, optional  Directory to save plots.
        rocky_only : bool           Restrict fits to rocky planets.

        Returns
        -------
        dict with keys: df, fit_a, fit_R, fit_2d, boundary, wind
        """
        df = self.load()

        if rocky_only:
            df_fit = df[df["planet_type"] == "rocky"].copy()
        else:
            df_fit = df.copy()

        fit_a    = self.fit_vs_distance(df_fit)
        fit_R    = self.fit_vs_radius(df_fit)
        fit_2d   = self.fit_2d_surface(df_fit)
        boundary = self.locking_boundary(fit_2d)
        wind     = self.wind_correlation(df_fit)

        self.print_summary(fit_a, fit_R, fit_2d, boundary)

        if plot_dir:
            try:
                import module_2_tidal_rotation.plots as plots
                os.makedirs(plot_dir, exist_ok=True)

                plots.plot_batch_locking_map(
                    df,
                    boundary=boundary,
                    save_path=os.path.join(plot_dir, "batch_locking_map.png")
                )
                plots.plot_empirical_fit(
                    fit_a, fit_R,
                    save_path=os.path.join(plot_dir, "empirical_fit.png")
                )
                plots.plot_wind_signature_population(
                    df_fit,
                    save_path=os.path.join(plot_dir, "wind_signature_population.png")
                )
            except Exception as e:
                print(f"  Warning — plot generation failed: {e}")

        return {
            "df":       df,
            "fit_a":    fit_a,
            "fit_R":    fit_R,
            "fit_2d":   fit_2d,
            "boundary": boundary,
            "wind":     wind,
        }
