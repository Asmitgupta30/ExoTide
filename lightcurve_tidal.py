"""
lightcurve_tidal.py  (Module 2 — Tidal Locking)
=================================================
Detects stellar rotation period from photometric lightcurve variability
(starspot modulation) and compares it to the orbital period to infer
the tidal locking state.

Method:
  Starspots on a rotating star modulate the total flux with a period equal
  to the stellar rotation period P_rot. If P_rot = P_orb, the star (and
  by tidal coupling, the planet) is or is approaching synchronous rotation.

  We extract P_rot using:
  1. Lomb-Scargle Periodogram (Zechmeister & Kurster 2009) — best for
     unevenly sampled data and stellar variability
  2. Autocorrelation Function (ACF) — alternative cross-check method
     (McQuillan et al. 2013)

  Data source: TESS/Kepler lightcurves via lightkurve (MAST archive)

References:
  - McQuillan et al. 2013, ApJ, 775, L11      (ACF rotation periods)
  - Zechmeister & Kurster 2009, A&A, 496, 577 (GLS periodogram)
  - Louden & Wheatley 2015, ApJ, 814, L24     (RM ingress/egress)
"""

import numpy as np
import warnings

try:
    import lightkurve as lk
except ImportError:
    lk = None


class LightcurveTidalAnalyser:
    """
    Extracts the stellar rotation period from lightcurve variability and
    infers the tidal locking state by comparing P_rot to P_orb.

    Usage
    -----
    lta = LightcurveTidalAnalyser()
    lc  = lta.download("TRAPPIST-1")
    rot = lta.find_rotation_period(lc, method="lomb_scargle")
    state = lta.assess_locking(rot["P_rot_days"], P_orb_days=6.1)
    lta.print_report(state)
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    def download(self, target, mission="TESS"):
        """
        Download a long-baseline lightcurve optimized for rotation period search.
        Uses all available sectors to capture multiple rotation cycles.
        """
        if lk is None:
            raise ImportError("lightkurve required. pip install lightkurve")
        print(f"[LightcurveTidalAnalyser] Downloading {mission} data for {target}...")
        sr = lk.search_lightcurve(target, mission=mission, author="SPOC")
        if len(sr) == 0:
            raise ValueError(f"No data found for {target}")
        # Download all sectors for maximum temporal baseline
        lc_col = sr.download_all()
        lc     = lc_col.stitch().remove_nans().remove_outliers(sigma=5)
        # Normalize but do NOT flatten — we want the rotation signal intact
        lc     = lc.normalize()
        print(f"  Downloaded {len(lc.time)} cadences over "
              f"{lc.time.value[-1] - lc.time.value[0]:.1f} days baseline.")
        return lc

    def load_local(self, filepath):
        """Load local .fits or .csv without flattening."""
        if lk is None:
            raise ImportError("lightkurve required.")
        if filepath.endswith((".fits", ".fit")):
            return lk.read(filepath).remove_nans().normalize()
        import pandas as pd
        df = pd.read_csv(filepath, comment="#")
        t  = df.iloc[:, 0].values
        f  = df.iloc[:, 1].values
        return lk.LightCurve(time=t, flux=f)

    # ------------------------------------------------------------------
    def find_rotation_period(self, lc, method="lomb_scargle",
                             min_period=0.5, max_period=100.0):
        """
        Extract the stellar rotation period from photometric variability.

        Parameters
        ----------
        lc         : LightCurve
        method     : str   "lomb_scargle" or "acf"
        min_period : float Search range minimum (days)
        max_period : float Search range maximum (days)

        Returns
        -------
        dict  P_rot_days, power_at_peak, method, secondary_peak_days
        """
        time = np.asarray(lc.time.value)
        flux = np.asarray(lc.flux.value)

        if method == "lomb_scargle":
            return self._lomb_scargle(time, flux, min_period, max_period)
        elif method == "acf":
            return self._acf_period(time, flux, min_period, max_period)
        else:
            raise ValueError("method must be 'lomb_scargle' or 'acf'")

    def _lomb_scargle(self, time, flux, min_p, max_p):
        from astropy.timeseries import LombScargle
        ls = LombScargle(time, flux)
        freq, power = ls.autopower(
            minimum_frequency=1.0/max_p,
            maximum_frequency=1.0/min_p
        )
        periods     = 1.0 / freq
        peak_idx    = np.argmax(power)
        P_rot       = float(periods[peak_idx])
        peak_power  = float(power[peak_idx])

        # Secondary peak (may indicate half-period harmonic from 2 spot groups)
        power2 = power.copy()
        half_w = int(0.05 * len(power))
        lo = max(0, peak_idx - half_w)
        hi = min(len(power), peak_idx + half_w)
        power2[lo:hi] = 0
        sec_idx = np.argmax(power2)
        P_secondary = float(periods[sec_idx])

        print(f"  [Lomb-Scargle] P_rot = {P_rot:.3f} days  "
              f"(power={peak_power:.4f}), secondary: {P_secondary:.3f} days")
        return {
            "P_rot_days":         P_rot,
            "power_at_peak":      peak_power,
            "secondary_peak_days": P_secondary,
            "method":             "Lomb-Scargle",
            "periods":            periods,
            "power":              power,
        }

    def _acf_period(self, time, flux, min_p, max_p):
        """
        Autocorrelation Function method (McQuillan et al. 2013).
        Measures the lag at which the lightcurve best correlates with itself.
        """
        dt      = float(np.median(np.diff(time)))
        acf     = np.correlate(flux - np.mean(flux),
                               flux - np.mean(flux), mode="full")
        acf     = acf[len(acf)//2:]
        acf     = acf / acf[0]
        lags    = np.arange(len(acf)) * dt

        # Find first significant peak past min_period
        search  = (lags >= min_p) & (lags <= max_p)
        if not np.any(search):
            return {"P_rot_days": np.nan, "method": "ACF", "error": "No period found"}

        lags_s  = lags[search]
        acf_s   = acf[search]
        peak_i  = np.argmax(acf_s)
        P_rot   = float(lags_s[peak_i])
        print(f"  [ACF] P_rot = {P_rot:.3f} days  (acf peak={acf_s[peak_i]:.4f})")
        return {
            "P_rot_days":    P_rot,
            "power_at_peak": float(acf_s[peak_i]),
            "method":        "ACF",
            "lags":          lags_s,
            "acf":           acf_s,
        }

    # ------------------------------------------------------------------
    def assess_locking(self, P_rot_days, P_orb_days,
                        system_age_gyr=None, tolerance=0.1):
        """
        Compare P_rot with P_orb to assess tidal locking state.

        Parameters
        ----------
        P_rot_days    : float  Stellar rotation period (days)
        P_orb_days    : float  Orbital period (days)
        system_age_gyr : float Optional — for context reporting
        tolerance     : float  Fractional tolerance for "locked" call

        Returns
        -------
        dict  is_locked, ratio, description, resonance_check
        """
        if np.isnan(P_rot_days) or P_rot_days <= 0:
            return {"is_locked": None, "description": "P_rot not determined."}

        ratio = P_rot_days / P_orb_days
        deviation = abs(ratio - 1.0)

        if deviation < tolerance:
            locked = True
            desc   = (f"P_rot ({P_rot_days:.3f}d) ~ P_orb ({P_orb_days:.3f}d). "
                      f"Consistent with 1:1 synchronous (tidally locked) rotation.")
        elif abs(ratio - 1.5) < tolerance:
            locked = False
            desc   = (f"P_rot/P_orb ~ 3:2 resonance (ratio={ratio:.3f}). "
                      f"Mercury-like spin-orbit resonance — not fully locked.")
        elif ratio < 0.5:
            locked = False
            desc   = (f"P_rot ({P_rot_days:.3f}d) << P_orb ({P_orb_days:.3f}d). "
                      f"Planet rotates much faster — not tidally locked.")
        else:
            locked = False
            desc   = (f"P_rot ({P_rot_days:.3f}d) / P_orb ({P_orb_days:.3f}d) = {ratio:.3f}. "
                      f"No clear resonance — freely rotating.")

        return {
            "is_locked":    locked,
            "P_rot_days":   P_rot_days,
            "P_orb_days":   P_orb_days,
            "ratio":        float(ratio),
            "description":  desc,
            "system_age_gyr": system_age_gyr,
        }

    def print_report(self, state):
        print("\n" + "="*60)
        print("  LIGHTCURVE TIDAL LOCKING ASSESSMENT")
        print("="*60)
        print(f"  P_rot:          {state.get('P_rot_days', 'N/A'):.3f} days")
        print(f"  P_orb:          {state.get('P_orb_days', 'N/A'):.3f} days")
        print(f"  P_rot / P_orb:  {state.get('ratio', 'N/A'):.4f}")
        print(f"  Locked:         {'YES' if state.get('is_locked') else 'NO'}")
        print(f"  Assessment:")
        print(f"    {state.get('description', '')}")
        print("="*60)
