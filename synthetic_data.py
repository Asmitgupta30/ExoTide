"""
synthetic_data.py  (Module 2 — Tidal Locking)
================================================
Generates synthetic Radial Velocity and transmission spectroscopy datasets
for a DEMONSTRATION planet that is NOT tidally locked.

The synthetic planet is placed just inside the tidal locking zone but
given a rotation period different from its orbital period, producing clear
atmospheric super-rotation signatures in the RV data.

Physical scenario:
  - Planet: 1.2 R_Earth, a = 0.15 AU, P_orb = 30 days, P_rot = 18 days
  - Star:   0.5 M_sun M-dwarf, T_eff = 3500 K
  - Result: tau_sync ~ 25 Gyr >> system age (5 Gyr) → NOT locked
  - Atmospheric state: strong equatorial super-rotating jet
    * Morning limb (ingress): receding from observer → REDSHIFT (+1.8 km/s)
    * Evening limb (egress):  approaching observer   → BLUESHIFT (-2.2 km/s)
    * Net disk-integrated blueshift: ~ -0.2 km/s (asymmetric)

This matches the observational signatures of HD 209458b (Snellen et al. 2010)
and HD 189733b (Louden & Wheatley 2015) but tuned for a smaller rocky planet.

References:
  - Snellen et al. 2010, Nature, 465, 1049
  - Louden & Wheatley 2015, ApJ, 814, L24
  - Showman & Polvani 2011, ApJ, 738, 71
"""

import numpy as np

# Physical constants
AU     = 1.496e11
G      = 6.674e-11
M_SUN  = 1.989e30
DAY_S  = 86400.0
C_KMS  = 2.998e5    # km/s

# Synthetic system parameters
SYNTH_PARAMS = {
    "name":            "SyntheticB (Non-Locked Demo)",
    "a_AU":            0.15,
    "M_star_Msun":     0.5,
    "R_planet_Rearth": 1.2,
    "P_orb_days":      30.0,
    "P_rot_days":      18.0,    # NOT equal to P_orb → not locked
    "eccentricity":    0.05,
    "system_age_gyr":  5.0,
    "v_ingress_km_s":  +1.8,    # Morning limb — redshift (wind away from us)
    "v_egress_km_s":   -2.2,    # Evening limb — blueshift (wind toward us)
    "v_err_km_s":      0.15,    # Measurement uncertainty (ESPRESSO-class)
    "v_bulk_planet_kms": 30.0,  # Orbital velocity of planet
    "transit_depth":   0.0025,  # (1.2 R_Earth / 0.5 R_sun)^2
    "transit_dur_hrs": 4.5,
}


class SyntheticTidalDataset:
    """
    Generates synthetic datasets for demonstrating a non-tidally-locked planet.

    Usage
    -----
    ds = SyntheticTidalDataset()
    rv_data = ds.generate_rv_timeseries()
    spec_data = ds.generate_transmission_spectra()
    ds.print_summary()
    """

    def __init__(self, params=None, seed=42):
        self.p = params or SYNTH_PARAMS.copy()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def generate_rv_timeseries(self, n_points=80):
        """
        Generate a synthetic Radial Velocity time-series spanning one transit.

        The RV signal contains:
          1. Keplerian orbital RV of the planet's star (sinusoidal baseline)
          2. Rossiter-McLaughlin anomaly during ingress/egress (asymmetric)
          3. Atmospheric wind contribution (asymmetric ingress vs egress shift)
          4. Gaussian measurement noise

        Returns
        -------
        dict with 'time_hrs', 'rv_km_s', 'rv_err_km_s', 'labels'
        """
        dur    = self.p["transit_dur_hrs"]
        t_mid  = 0.0
        # Time array: 2x transit duration centered on mid-transit
        t_hrs  = np.linspace(-dur, dur, n_points)

        # Normalized ingress/egress phases
        t_in_start  = -dur / 2.0
        t_in_end    = t_in_start + dur * 0.15   # ingress takes 15% of duration
        t_eg_start  = dur / 2.0 - dur * 0.15
        t_eg_end    = dur / 2.0

        rv = np.zeros(n_points)

        # RM anomaly contribution (planet blocking the spinning star)
        # Blue-shifts then red-shifts as planet crosses the disk
        # For this non-locked planet the RM is also asymmetric due to diff rotation
        for i, t in enumerate(t_hrs):
            if t_in_start <= t <= t_in_end:
                # Ingress: planet moves across the blue-shifted (approaching) limb
                frac = (t - t_in_start) / (t_in_end - t_in_start)
                rv[i] = -0.8 * frac              # RM anomaly: blueshift
            elif t_in_end < t < t_eg_start:
                # Full transit: planet crossing star center — RM reversal
                frac = (t - t_in_end) / (t_eg_start - t_in_end)
                rv[i] = -0.8 + 1.6 * frac        # crosses zero → redshift
            elif t_eg_start <= t <= t_eg_end:
                # Egress: planet crosses the red-shifted (receding) limb
                frac = (t - t_eg_start) / (t_eg_end - t_eg_start)
                rv[i] = 0.8 * (1 - frac)         # RM dies off

        # Add atmospheric wind shift:
        # During ingress, the morning limb blows AWAY → additional redshift
        # During egress, the evening limb blows TOWARD → additional blueshift
        for i, t in enumerate(t_hrs):
            if t_in_start <= t <= t_in_end:
                rv[i] += self.p["v_ingress_km_s"] * 0.4  # atmospheric redshift
            elif t_eg_start <= t <= t_eg_end:
                rv[i] += self.p["v_egress_km_s"] * 0.4   # atmospheric blueshift

        # Measurement noise (variable per point)
        base_noise = self.p["v_err_km_s"]
        rv_err = base_noise * (1.0 + 0.3 * self.rng.uniform(-1, 1, n_points))
        noise = self.rng.normal(0, rv_err)
        rv_obs = rv + noise

        # Labels for each point
        labels = []
        for t in t_hrs:
            if t_in_start <= t <= t_in_end:
                labels.append("ingress")
            elif t_in_end < t < t_eg_start:
                labels.append("in-transit")
            elif t_eg_start <= t <= t_eg_end:
                labels.append("egress")
            else:
                labels.append("out-of-transit")

        return {
            "time_hrs":    t_hrs,
            "rv_km_s":     rv_obs,
            "rv_true_km_s": rv,
            "rv_err_km_s": rv_err,
            "labels":      labels,
            "v_ingress":   self.p["v_ingress_km_s"],
            "v_egress":    self.p["v_egress_km_s"],
            "t_ingress":   (t_in_start, t_in_end),
            "t_egress":    (t_eg_start, t_eg_end),
        }

    # ------------------------------------------------------------------
    def generate_transmission_spectra(self, n_pixels=500, line_center_nm=589.0):
        """
        Generate synthetic ingress vs. egress transmission spectra showing
        the Doppler shifts from atmospheric super-rotation.

        The sodium doublet line at 589 nm is used as the reference.
        Ingress spectrum (morning limb): line center shifted to longer lambda (redshift)
        Egress spectrum (evening limb):  line center shifted to shorter lambda (blueshift)

        Returns
        -------
        dict with 'wavelength_nm', 'flux_ingress', 'flux_egress',
                  'delta_v_km_s', 'line_center_nm'
        """
        v_in = self.p["v_ingress_km_s"]   # redshift (+)
        v_eg = self.p["v_egress_km_s"]    # blueshift (-)

        # Doppler shift in nm: delta_lambda = lambda * v/c
        dl_in = line_center_nm * v_in / C_KMS   # positive → redshifted
        dl_eg = line_center_nm * v_eg / C_KMS   # negative → blueshifted

        wavelengths = np.linspace(line_center_nm - 2.0, line_center_nm + 2.0, n_pixels)
        sigma = 0.12   # line width in nm (instrumental + thermal broadening)

        # Gaussian absorption line depth ~ 0.15
        def absorption_line(wl, center, depth=0.15):
            return 1.0 - depth * np.exp(-0.5 * ((wl - center) / sigma)**2)

        flux_in = absorption_line(wavelengths, line_center_nm + dl_in)
        flux_eg = absorption_line(wavelengths, line_center_nm + dl_eg)

        # Add photon noise
        flux_in += self.rng.normal(0, 0.008, n_pixels)
        flux_eg += self.rng.normal(0, 0.008, n_pixels)

        delta_v = v_in - v_eg   # total ingress-egress velocity separation

        return {
            "wavelength_nm":  wavelengths,
            "flux_ingress":   flux_in,
            "flux_egress":    flux_eg,
            "line_center_nm": line_center_nm,
            "dl_ingress_nm":  dl_in,
            "dl_egress_nm":   dl_eg,
            "delta_v_km_s":   delta_v,
            "v_ingress_km_s": v_in,
            "v_egress_km_s":  v_eg,
        }

    # ------------------------------------------------------------------
    def print_summary(self):
        p = self.p
        print("\n" + "="*60)
        print("  SYNTHETIC DATASET — Non-Tidally-Locked Planet")
        print("="*60)
        print(f"  System:             {p['name']}")
        print(f"  Orbital semi-major: {p['a_AU']} AU")
        print(f"  Orbital period:     {p['P_orb_days']} days")
        print(f"  Rotation period:    {p['P_rot_days']} days  (NOT = P_orb!)")
        print(f"  Eccentricity:       {p['eccentricity']}")
        print(f"  System age:         {p['system_age_gyr']} Gyr")
        print(f"\n  Atmospheric Wind Signature:")
        print(f"    Ingress (morning limb):  v = {p['v_ingress_km_s']:+.2f} km/s  "
              f"[REDSHIFT — wind receding]")
        print(f"    Egress  (evening limb):  v = {p['v_egress_km_s']:+.2f} km/s  "
              f"[BLUESHIFT — wind approaching]")
        dv = p['v_ingress_km_s'] - p['v_egress_km_s']
        print(f"    Delta_v:                 {dv:+.2f} km/s  → Super-rotation JET confirmed")
        print(f"    Net disk blueshift:    "
              f"  {(p['v_ingress_km_s']+p['v_egress_km_s'])/2:+.2f} km/s")
        print(f"\n  Data sources for REAL systems:")
        print(f"    Photometric RV: HARPS (ESO Archive) / ESPRESSO / HIRES")
        print(f"    Transmission:   CARMENES / CRIRES+ / SPIRou")
        print("="*60)
