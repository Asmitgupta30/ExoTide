# Module 2 — Rotational and Tidal Locking Analyser

This module provides a complete analytical framework for determining whether an exoplanet is tidally locked, deriving its rotation state from first principles, detecting atmospheric super-rotation, and fitting the Rossiter-McLaughlin effect from Radial Velocity data using MCMC.

---

## Contents

| File | Description |
|---|---|
| `rotation_analyser.py` | Tidal sync timescale, pseudo-synchronous rotation (Hut 1981), resonance detection |
| `rm_solver.py` | Rossiter-McLaughlin analytical model + MCMC posterior sampling |
| `lightcurve_tidal.py` | Stellar rotation period extraction from photometric variability |
| `synthetic_data.py` | Synthetic RV + transmission spectra for a non-locked planet demonstration |
| `plots.py` | RM anomaly, transmission spectra, sensitivity heatmap, corner plots |
| `run.py` | Jupyter-friendly entry point with TIDAL_MODE and RUN_SYNTHETIC switches |

---

## How to Run

Open `run.py` and configure the switches at the top. Then in Jupyter:

    %run module_2_tidal_rotation/run.py

Or in terminal:

    python module_2_tidal_rotation/run.py

The key switches are:

    TIDAL_MODE   = "mcmc"        # "lightcurve" or "mcmc"
    TARGET       = "TRAPPIST-1"  # MAST target name or local file path
    RUN_SYNTHETIC = True          # Generate synthetic non-locked planet demo

---

## Where Does Actual Data Come From?

This is the most important practical distinction between the two modes.

### Lightcurve Mode — Photometry from TESS/Kepler

The stellar rotation period is extracted from brightness modulations caused by starspots rotating in and out of view. The data source is:

- TESS (Transiting Exoplanet Survey Satellite): Observed by NASA, raw cadence data hosted at:
  https://mast.stsci.edu/
  Downloaded automatically via lightkurve using `search_lightcurve(target, mission="TESS")`.

- Kepler / K2: Use `mission="Kepler"` or `mission="K2"` in the search call.

This mode does NOT measure the planet's rotation directly. It measures the stellar rotation period P_rot and compares it to the orbital period P_orb. If P_rot ≈ P_orb, the system is in or approaching synchronous rotation.

### MCMC Mode — Radial Velocity from Ground Spectrographs

The Rossiter-McLaughlin effect requires time-resolved spectroscopic Radial Velocity measurements taken during a planetary transit. This data is NOT available from TESS. It requires ground-based high-resolution spectrographs.

The primary public archives for real RM data are:

HARPS (High Accuracy Radial velocity Planet Searcher):
- Instrument: 3.6m ESO telescope, La Silla, Chile
- Precision: approximately 1 m/s
- Archive: http://archive.eso.org/
- Search: use target name and filter Phase III data, select HARPS spectra or reduced RV files

ESPRESSO (Echelle Spectrograph for Rocky Exoponents and Stable Spectroscopic Observations):
- Instrument: VLT 8m, Paranal, Chile
- Precision: approximately 0.1 m/s (best in the world)
- Archive: http://archive.eso.org/
- Search: same ESO portal, select ESPRESSO

HIRES (High Resolution Echelle Spectrometer):
- Instrument: Keck 10m, Mauna Kea, Hawaii
- Precision: approximately 1-3 m/s
- Archive: https://koa.ipac.caltech.edu/
- Format: calibrated 1D FITS spectra, often also pre-reduced RV time series

NEID (NN-EXPLORE Exoplanet Investigations with Doppler spectroscopy):
- Instrument: WIYN 3.5m, Kitt Peak National Observatory
- Precision: approximately 0.3 m/s
- Archive: https://neid.ipac.caltech.edu/

### Transmission Spectra — Atmospheric Wind Detection

To observationally prove super-rotation (atmospheric Doppler shift), high-resolution transmission spectroscopy is required:

CARMENES (Calar Alto high-Resolution search for M dwarfs with Exoearths with Near-infrared and optical Echelle Spectrographs):
- Best for M dwarfs in the near-infrared
- Archive: https://carmenes.cab.inta-csic.es/

CRIRES+ (CRyogenic high-resolution InfraRed Echelle Spectrograph):
- ESO VLT, excellent for CO, H2O, CH4 detection
- Archive: http://archive.eso.org/

SPIRou (SpectroPolarimetre InfraRouge):
- CFHT 3.6m, Mauna Kea
- Optimized for cool star metallicity and wind detection

In practice for a term project, we use the synthetic data generator in `synthetic_data.py` which produces physically motivated RV and spectra datasets with the correct ingress/egress asymmetries.

---

## Physical Framework

### 1. Tidal Synchronization Timescale

The time required for tidal dissipation to force a planet into synchronous rotation is derived from the Gladman et al. (1996) formulation of the Goldreich and Soter (1966) tidal theory:

    tau_sync = (2 * pi * xi / 9) * omega_0 * M_p * Q * a^6
               / (G * k2 * M_*^2 * R_p^3)

where:
- xi = 0.33 for rocky planets, 0.25 for gas/ice giants (moment of inertia factor)
- omega_0 = initial spin rate (taken as Earth's rotation rate unless specified)
- M_p = planet mass (kg), estimated via the Chen and Kipping (2017) mass-radius relation
- Q = tidal dissipation factor (rocky: 100, ice giant: 1000, gas giant: 10^5)
- k2 = second-order Love number (rocky: 0.299, gas giant: 0.37)
- a = semi-major axis (m)
- G = gravitational constant
- M_* = stellar mass (kg)
- R_p = planet radius (m)

The planet is considered locked if the system age t_age is greater than tau_sync:

    t_age / tau_sync > 1 -> likely locked
    t_age / tau_sync > 10 -> very likely locked with high confidence

A sensitivity grid over Q and k2 parameter space is generated to quantify the probability across plausible physical scenarios.


### 2. Pseudo-Synchronous Rotation (Eccentric Orbits)

For planets in eccentric orbits (e > 0.01), the equilibrium is not synchronous rotation. Instead, the planet reaches a pseudo-synchronous state where the rotation rate is faster than the mean orbital motion, defined by the Hut (1981) formula:

    omega_ps / n = f(e^2, e^4, e^6)

Specifically:

    omega_ps / n = (1 + (15/2)*e^2 + (45/8)*e^4 + (5/16)*e^6)
                   / ((1 + 3*e^2 + (3/8)*e^4) * (1 - e^2)^(3/2))

where n = 2*pi/P_orb is the mean orbital angular velocity. For e = 0 this reduces exactly to omega_ps = n (fully synchronous). For moderate eccentricities (e ~ 0.1) the pseudo-synchronous rate can be 10-30% faster than the orbital mean motion.


### 3. Spin-Orbit Resonances

Even when not in 1:1 synchronous lock, a planet may be trapped in a higher-order resonance. The module checks for the following discrete states:

    omega_rot / n = p/q

where (p, q) are:
- (1, 1): fully synchronous (tidally locked)
- (3, 2): Mercury-type resonance
- (2, 1): fast rotator
- (1, 2): sub-synchronous

A match within 5% of the exact ratio is flagged as a probable resonance.


### 4. Rossiter-McLaughlin Effect

When a planet transits across the face of a spinning star, it sequentially blocks the blueshifted (approaching) and redshifted (receding) hemispheres of the stellar disk. This produces a characteristic anomalous Radial Velocity signal during the transit.

The maximum semi-amplitude of the anomaly is (Cristo et al. 2024):

    Delta_v_RM = v * sin(i) * (Rp/R_*)^2 * sqrt(1 - b^2)

The time-resolved profile is modeled using the Ohta, Taruya, and Suto (2005) analytical formulation. The planet position projected onto the stellar equatorial coordinate system is:

    x_c(t) = (a/R_*) * sin(2*pi*phase(t))
    y_c(t) = b

Rotated by the spin-orbit angle lambda:

    x_rot = x_c * cos(lambda) - y_c * sin(lambda)

The local RV anomaly is proportional to the x_rot coordinate (the position along the stellar rotation axis), weighted by the fraction of disk area blocked:

    delta_v(t) = v*sin(i) * x_rot(t) * (Rp/R_*)^2 * ingress_damping(t)

MCMC samples the two-parameter space [v_sin_i, lambda] with:
- Flat prior on v_sin_i in [0, 200] km/s
- Flat prior on lambda in [-pi, pi] rad
- Gaussian log-likelihood against the observed RV time series

The spin-orbit angle lambda is the primary diagnostic:
- |lambda| < 20 deg: aligned orbit, consistent with tidal synchronization
- 20 < |lambda| < 60 deg: moderate misalignment, possible atmospheric drag
- |lambda| > 60 deg: severely misaligned or retrograde orbit


### 5. Atmospheric Super-Rotation Detection

For a tidally locked planet, the permanent day-side heating drives a strong equatorial super-rotating jet. This jet advects heat from the sub-stellar point eastward, warming the evening (eastern) limb relative to the morning (western) limb.

From a spectroscopic perspective:
- During ingress the morning limb is visible. The jet blows material away from the observer. This produces a net REDSHIFT of atmospheric absorption lines.
- During egress the evening limb is visible. The jet blows material toward the observer. This produces a net BLUESHIFT.

The ingress-egress velocity asymmetry is:

    Delta_v_atm = v_ingress - v_egress > 0   for prograde super-rotation

This was first observed in HD 209458b where Snellen et al. (2010) measured a disk-integrated blueshift of approximately 2 km/s in CO lines using CRIRES, and confirmed in HD 189733b by Louden and Wheatley (2015) with an ingress-egress asymmetry of 1.7 km/s using HARPS.

The synthetic dataset in `synthetic_data.py` reproduces this exact physical scenario for a rocky planet with:
- v_ingress = +1.8 km/s (morning limb redshift)
- v_egress = -2.2 km/s (evening limb blueshift)
- Delta_v = 4.0 km/s
- Net disk-integrated blueshift: -0.2 km/s

A planet that is NOT tidally locked would show a much smaller and more symmetric ingress-egress velocity pattern, because the rotation is quicker and the thermal contrast between the two limbs is weaker.

---

## Output Plots

All plots are saved to `output_plots/module2/`:

- `tidal_sensitivity_heatmap.png` — Q-k2 grid showing locked/not-locked across plausible parameter space
- `rm_anomaly.png` — Fitted Rossiter-McLaughlin RV profile with tidal state annotation
- `rm_corner.png` — MCMC posterior corner plot for [v_sin_i, lambda]
- `synthetic_unlocked_rv.png` — Synthetic RV time series showing the asymmetric ingress-egress signal
- `synthetic_transmission_spectra.png` — Ingress vs. egress absorption line Doppler shift

---

## References

Gladman, B., Dane Quinn, D., Nicholson, P., and Rand, R. (1996). Synchronous Locking of Tidally Evolving Satellites. Icarus, 122(1), 166-192.

Goldreich, P. and Soter, S. (1966). Q in the Solar System. Icarus, 5(1-6), 375-389.

Hut, P. (1981). Tidal evolution in close binary systems. Astronomy and Astrophysics, 99, 126-140.

Murray, C. D. and Dermott, S. F. (1999). Solar System Dynamics. Cambridge University Press.

Ohta, Y., Taruya, A., and Suto, Y. (2005). The Rossiter-McLaughlin Effect and Analytic Radial Velocity Curves for Transiting Extrasolar Planetary Systems. The Astrophysical Journal, 622(2), 1118-1135.

Cristo, E., et al. (2024). The Rossiter-McLaughlin effect and the JWST connection. Astronomy and Astrophysics, 683, A227.

Espinoza-Retamal, J. I., et al. (2024). Spin-orbit angles of transiting exoplanets. Monthly Notices of the Royal Astronomical Society, 532(2).

Foreman-Mackey, D., Hogg, D. W., Lang, D., and Goodman, J. (2013). emcee: The MCMC Hammer. Publications of the Astronomical Society of the Pacific, 125(925), 306.

Chen, J. and Kipping, D. (2017). Probabilistic Forecasting of the Masses and Radii of Other Worlds. The Astrophysical Journal, 834(1), 17.

Snellen, I. A. G., et al. (2010). The orbital motion, absolute mass and high-altitude winds of exoplanet HD 209458b. Nature, 465(7301), 1049-1051.

Louden, T. and Wheatley, P. J. (2015). Spatially Resolved Eastward Winds and Rotation of HD 189733b. The Astrophysical Journal, 814(2), L24.

Showman, A. P. and Polvani, L. M. (2011). Equatorial Superrotation on Tidally Locked Exoplanets. The Astrophysical Journal, 738(1), 71.

McQuillan, A., Aigrain, S., and Mazeh, T. (2013). Measuring the rotation period distribution of field M dwarfs with Kepler. Monthly Notices of the Royal Astronomical Society, 432(2), 1203-1216.

Zechmeister, M. and Kurster, M. (2009). The generalised Lomb-Scargle periodogram. Astronomy and Astrophysics, 496(2), 577-584.
