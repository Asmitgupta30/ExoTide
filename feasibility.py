"""
feasibility.py
==============
Detection feasibility analysis for atmospheric Doppler spin-state
measurements on HWO target planets.

The core question:
  Given a planet with predicted Doppler asymmetry Dv_total (km/s),
  how many transits does HWO (or another telescope) need to detect
  the spin state at 5-sigma significance?

Physics
-------
The Doppler centroid precision for a high-resolution spectrograph:

    sigma_v = (c / R) / (sqrt(N_lines) * SNR_per_reselem)

where:
    c          = speed of light (km/s)
    R          = spectral resolving power
    N_lines    = number of useful molecular lines in the bandpass
    SNR_per_reselem = sqrt(N_photons per resolution element)

For detection: Dv_total / sigma_v > 5  (5-sigma threshold)

Required SNR:
    SNR_req = 5 * (c/R) / (sqrt(N_lines) * Dv_total)

Photon budget:
    N_photons proportional to  telescope_area * transit_duration * stellar_flux

Stellar flux in K-band (2.2 um, where CO lines are for warm planets):
    F_K ~ L_K(Teff) / (4*pi*d^2)
    approximated via blackbody peak + bolometric correction

References:
    Rodler & Lopez-Morales (2014) ApJ 781, 54
    Snellen et al. (2015) A&A 576, A59
    Marconi et al. (2022) SPIE 12184, ELT-HIRES specs
"""

import numpy as np
import math

# ─────────────────────────────────────────────────────────────────────────────
# Instrument presets
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "HWO_6m": {
        "diameter_m":      6.0,
        "resolving_power": 100_000,
        "throughput":      0.12,      # end-to-end incl. atmosphere
        "n_lines_rocky":   15,        # H2O + CO lines, rocky/warm planet
        "n_lines_giant":   60,        # more lines for gas giant
        "wavelength_um":   1.6,       # H-band (H2O)
        "label":           "HWO 6m",
    },
    "ELT_HIRES": {
        "diameter_m":      39.0,
        "resolving_power": 100_000,
        "throughput":      0.10,
        "n_lines_rocky":   20,
        "n_lines_giant":   80,
        "wavelength_um":   1.6,
        "label":           "ELT-HIRES 39m",
    },
    "VLT_CRIRES": {
        "diameter_m":      8.2,
        "resolving_power": 100_000,
        "throughput":      0.08,
        "n_lines_rocky":   12,
        "n_lines_giant":   50,
        "wavelength_um":   2.3,       # K-band CO
        "label":           "VLT/CRIRES+ 8.2m",
    },
}

C_KMS     = 2.998e5     # km/s
H_PLANCK  = 6.626e-34   # J s
K_BOLTZ   = 1.381e-23   # J/K
AU_PC     = 206265.0    # AU per parsec


# ─────────────────────────────────────────────────────────────────────────────
def stellar_flux_photons(Teff_K, distance_pc, diameter_m, wavelength_um,
                          throughput, transit_duration_hr):
    """
    Estimate photon count per resolution element during one transit.

    Uses a blackbody approximation at wavelength_um.  Real spectrographs
    would use a stellar model, but this is sufficient for an order-of-magnitude
    feasibility calculation.

    Parameters
    ----------
    Teff_K            : float   Stellar effective temperature (K)
    distance_pc       : float   Distance to the star (parsec)
    diameter_m        : float   Telescope primary diameter (m)
    wavelength_um     : float   Observing wavelength (micron)
    throughput        : float   System throughput (0-1)
    transit_duration_hr: float  Transit duration (hours)

    Returns
    -------
    float  Number of photons collected per resolution element during transit
    """
    lam_m     = wavelength_um * 1e-6     # wavelength in metres
    lam_hz    = C_KMS * 1e3 / lam_m     # frequency (Hz)
    delta_lam = lam_m / 100_000         # width of one resolution element (R=1e5)
    delta_hz  = C_KMS * 1e3 * delta_lam / lam_m**2

    # Planck function  B_nu (W / m^2 / Hz / sr)
    x       = H_PLANCK * lam_hz / (K_BOLTZ * Teff_K)
    B_nu    = (2.0 * H_PLANCK * lam_hz**3 / C_KMS**3 / 1e9) / (np.exp(x) - 1.0)
    # Solid angle of stellar disk (approximate using main-sequence R-M relation)
    R_star_Rsun = (Teff_K / 5778.0) ** 1.5     # very rough
    R_star_m    = R_star_Rsun * 6.957e8
    d_m         = distance_pc * 3.086e16
    omega_star  = np.pi * (R_star_m / d_m)**2   # sr

    # Flux density at telescope (W / m^2 / Hz)
    F_nu = B_nu * omega_star * np.pi     # integrate over stellar disk (pi sr factor)

    # Photons per res-element during transit
    A_tel       = np.pi * (diameter_m / 2.0)**2
    t_obs_s     = transit_duration_hr * 3600.0
    E_photon    = H_PLANCK * lam_hz
    N_photons   = (F_nu * delta_hz * A_tel * t_obs_s * throughput) / E_photon

    return max(float(N_photons), 1.0)


def sigma_v(instrument_key, n_photons, R_planet_Rearth):
    """
    Doppler velocity precision for one transit.

    sigma_v = (c/R) / (sqrt(N_lines) * SNR)

    SNR = sqrt(N_photons_per_reselem)
    N_lines depends on planet type (rocky vs giant)
    """
    inst    = INSTRUMENTS[instrument_key]
    R       = inst["resolving_power"]
    n_lines = inst["n_lines_rocky"] if R_planet_Rearth < 5.0 else inst["n_lines_giant"]
    snr     = np.sqrt(max(n_photons, 1.0))
    return C_KMS / (R * np.sqrt(n_lines) * snr)


def transits_needed(dv_total_kms, sigma_v_per_transit, detection_sigma=5.0):
    """
    Number of transits required for detection at given sigma level.

    sigma_v improves as 1/sqrt(N_transits), so:
        N_transits = (detection_sigma * sigma_v_per_transit / dv_total)^2

    Capped at 200 to flag "impractical".
    """
    if dv_total_kms <= 0.0:
        return 999
    n = (detection_sigma * sigma_v_per_transit / dv_total_kms) ** 2
    return int(min(math.ceil(n), 999))


# ─────────────────────────────────────────────────────────────────────────────
class DetectionFeasibilityAnalyser:
    """
    Compute and rank HWO/ELT detection feasibility for all planets in
    the batch results DataFrame.

    Usage
    -----
    dfa = DetectionFeasibilityAnalyser(df_batch_with_doppler)
    df_feas = dfa.run(instrument="HWO_6m")
    dfa.print_priority_table(df_feas)
    """

    # Typical transit durations by planet type (rough scaling with a^0.5)
    T_TRANSIT_FRAC = 0.01   # transit duration / orbital period (geometric mean)

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame  Must contain columns from batch_processor output
                           including dv_total_kms, a_AU, P_orb_days,
                           R_planet_Rearth, M_star_Msun, Teff_K, distance_pc
        """
        self.df = df.copy()

    @staticmethod
    def estimate_distance_pc(M_star_Msun, Teff_K, apparent_mag_V=6.0):
        """
        Very rough distance estimate from stellar luminosity and apparent mag.
        For most HWO targets the distance is known; this is a fallback.

        L ~ M^4 (mass-luminosity), absolute_V from solar calibration.
        """
        L_solar = M_star_Msun ** 3.5
        M_V     = 4.83 - 2.5 * np.log10(max(L_solar, 0.001))
        dist_pc = 10 ** ((apparent_mag_V - M_V + 5) / 5.0)
        return float(np.clip(dist_pc, 0.5, 1000.0))

    @staticmethod
    def transit_duration_hr(a_AU, P_orb_days, M_star_Msun, R_planet_Rearth,
                             b=0.3):
        """
        Geometric transit duration (Winn 2010):
            T_14 = (P/pi) * arcsin( R_* / a * sqrt((1+Rp/R*)^2 - b^2) )

        Uses Demircan & Kahraman (1991) stellar radius from mass.
        """
        R_SUN_IN_REARTH = 109.076
        R_star_Rsun     = M_star_Msun ** 0.80
        R_star_Rearth   = R_star_Rsun * R_SUN_IN_REARTH
        p_ratio         = R_planet_Rearth / R_star_Rearth
        AU_IN_REARTH    = 1.496e11 / 6.371e6

        a_Rearth = a_AU * AU_IN_REARTH
        arg      = (R_star_Rearth / a_Rearth) * np.sqrt(
            max((1 + p_ratio)**2 - b**2, 0.0)
        )
        arg      = float(np.clip(arg, 0.0, 1.0))
        T_days   = (P_orb_days / np.pi) * np.arcsin(arg)
        return float(T_days * 24.0)    # hours

    def run(self, instrument="HWO_6m", detection_sigma=5.0):
        """
        Compute feasibility for every planet.

        Adds columns:
            T_transit_hr       Transit duration (hours)
            N_photons_transit  Photon count per transit per res-elem
            sigma_v_kms        Doppler velocity precision per transit (km/s)
            N_transits_HWO     Transits needed for 5-sigma detection
            feasible_10        True if detectable in <= 10 transits
            priority_score     1/N_transits (higher = easier target)
            instrument         Instrument label used

        Returns
        -------
        pd.DataFrame  Sorted by priority_score descending
        """
        inst    = INSTRUMENTS[instrument]
        records = []

        for _, row in self.df.iterrows():
            pname   = str(row.get("planet_name", "?"))
            a       = float(row.get("a_AU", 1.0))
            P_days  = float(row.get("P_orb_days", 365.0))
            R_p     = float(row.get("R_planet_Rearth", 1.0))
            M_s     = float(row.get("M_star_Msun", 1.0))
            Teff    = float(row.get("Teff_K", 5778.0))
            dv      = float(row.get("dv_total_kms", 0.0))
            d_pc    = float(row.get("distance_pc",
                                    self.estimate_distance_pc(M_s, Teff)))

            T_tr_hr = self.transit_duration_hr(a, P_days, M_s, R_p)

            N_ph = stellar_flux_photons(
                Teff, d_pc,
                inst["diameter_m"],
                inst["wavelength_um"],
                inst["throughput"],
                T_tr_hr
            )

            sv      = sigma_v(instrument, N_ph, R_p)
            N_tr    = transits_needed(dv, sv, detection_sigma)
            score   = 1.0 / max(N_tr, 1)

            records.append({
                "planet_name":        pname,
                "dv_total_kms":       round(dv, 4),
                "T_transit_hr":       round(T_tr_hr, 3),
                "N_photons_transit":  int(N_ph),
                "sigma_v_kms":        round(sv, 5),
                "N_transits_needed":  N_tr,
                "feasible_10":        N_tr <= 10,
                "priority_score":     round(score, 5),
                "instrument":         inst["label"],
                "distance_pc":        round(d_pc, 2),
            })

        import pandas as pd
        df_out = pd.DataFrame(records).sort_values(
            "priority_score", ascending=False
        ).reset_index(drop=True)
        df_out["priority_rank"] = range(1, len(df_out) + 1)
        return df_out

    @staticmethod
    def print_priority_table(df_feas, top_n=20):
        """Print the top-N planets as a formatted priority table."""
        print("\n" + "="*78)
        print("  DETECTION FEASIBILITY RANKING  (top {})".format(top_n))
        print("="*78)
        print(f"  {'Rank':<5} {'Planet':<22} {'Dv (km/s)':<12} "
              f"{'sigma_v':<12} {'N_transits':<12} {'Feasible?'}")
        print("-"*78)
        for _, row in df_feas.head(top_n).iterrows():
            feas = "YES" if row["feasible_10"] else "no"
            print(f"  {int(row['priority_rank']):<5} "
                  f"{str(row['planet_name']):<22} "
                  f"{row['dv_total_kms']:<12.4f} "
                  f"{row['sigma_v_kms']:<12.5f} "
                  f"{int(row['N_transits_needed']):<12} "
                  f"{feas}")
        print("="*78)
