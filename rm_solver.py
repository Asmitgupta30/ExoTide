"""
rm_solver.py  (Module 2 — Tidal Locking)
==========================================
Extended Rossiter-McLaughlin effect solver.

Improvements over the parent rm_solver.py:
  - MCMC sampling for v_sin_i and lambda (full posterior distributions,
    not just point estimates from scipy curve_fit)
  - Analytical Ohta-Taruya-Suto (OTS) kinematic forward model
  - Spin-orbit obliquity interpretation with tidal locking designation

Physical model:
  The RM anomaly amplitude is (Cristo et al. 2024):
    Delta_v = v*sin(i) * (Rp/R*)^2 * sqrt(1 - b^2)

  The signed shape is determined by the spin-orbit angle lambda:
    lambda = 0  → aligned (prograde orbit)
    lambda = pi → retrograde orbit
    lambda > 20 deg → possible atmospheric super-rotation drag

References:
  - Ohta, Taruya & Suto 2005, ApJ, 622, 1118     (OTS RM model)
  - Cristo et al. 2024, A&A, 683, A227            (RM amplitude formula)
  - Espinoza-Retamal et al. 2024, MNRAS, 532, 2   (obliquity catalogue)
  - Foreman-Mackey et al. 2013, PASP, 125, 306    (emcee sampler)

Data Sources for Real RM Observations:
  - HARPS (ESO public archive):     http://archive.eso.org/
  - ESPRESSO:                       http://archive.eso.org/
  - HIRES (Keck Observatory):       https://koa.ipac.caltech.edu/
  - NEID:                           https://neid.ipac.caltech.edu/
  These archives provide calibrated 1D spectra or reduced RV time series.
  Search by target name and filter by "in-transit" observation windows.
"""

import numpy as np
import emcee
from scipy.optimize import curve_fit


class RMSolverMCMC:
    """
    Rossiter-McLaughlin solver with full MCMC posterior sampling.

    Usage
    -----
    rm = RMSolverMCMC()
    profile = rm.ohta_rm_profile(t, t0, period, v_sin_i, lam, p, a, b)
    results, samples = rm.run_mcmc(t_rv, rv_data, rv_err, p, a, b, t0, period)
    """

    def __init__(self, nwalkers=32, nsteps=2000, nburn=400, nthin=10):
        self.nwalkers = nwalkers
        self.nsteps   = nsteps
        self.nburn    = nburn
        self.nthin    = nthin

    # ------------------------------------------------------------------
    @staticmethod
    def theoretical_amplitude(v_sin_i, p_ratio, b):
        """
        Cristo et al. (2024) maximum RM semi-amplitude.
        Delta_v = v*sin(i) * (Rp/R*)^2 * sqrt(1 - b^2)
        """
        b_safe = min(abs(b), 0.999)
        return v_sin_i * (p_ratio**2) * np.sqrt(1.0 - b_safe**2)

    # ------------------------------------------------------------------
    @staticmethod
    def ohta_rm_profile(t, t0, period, v_sin_i, lam, p_ratio, a_ratio, b):
        """
        OTS analytical Radial Velocity anomaly time series.

        The planet blocks a patch of the stellar disk with local velocity
        v_local = v_sin_i * x_rot / R_*
        where x_rot is the planet position projected onto the rotation axis.

        Parameters
        ----------
        t       : array   Time array (same units as period/t0)
        t0      : float   Mid-transit epoch
        period  : float   Orbital period
        v_sin_i : float   Projected stellar equatorial velocity (km/s)
        lam     : float   Spin-orbit angle (radians); 0=aligned, pi=retrograde
        p_ratio : float   Rp/R_*
        a_ratio : float   a/R_*
        b       : float   Impact parameter

        Returns
        -------
        ndarray   RV anomaly in same units as v_sin_i
        """
        phase = (t - t0) / period
        phase = phase - np.floor(phase + 0.5)
        angle = 2.0 * np.pi * phase

        x_c = a_ratio * np.sin(angle)
        y_c = b
        x_rot = x_c * np.cos(lam) - y_c * np.sin(lam)

        z = np.sqrt(x_c**2 + y_c**2)
        anomaly = np.zeros_like(t, dtype=float)
        in_transit = z < (1.0 + p_ratio)

        if np.any(in_transit):
            blocked_frac = p_ratio**2
            edge_damp = np.clip(
                (1.0 + p_ratio - z[in_transit]) / (2.0 * p_ratio), 0.0, 1.0
            )
            anomaly[in_transit] = (v_sin_i * x_rot[in_transit]
                                   * blocked_frac * edge_damp)
        return anomaly

    # ------------------------------------------------------------------
    def _log_prior(self, theta):
        """Priors on [v_sin_i, lambda]."""
        v_sin_i, lam = theta
        if 0.0 < v_sin_i < 200.0 and -np.pi < lam < np.pi:
            return 0.0
        return -np.inf

    def _log_likelihood(self, theta, t, rv, rv_err, p, a, b, t0, period):
        v_sin_i, lam = theta
        model  = self.ohta_rm_profile(t, t0, period, v_sin_i, lam, p, a, b)
        sigma2 = rv_err**2
        return -0.5 * np.sum((rv - model)**2 / sigma2 + np.log(sigma2))

    def _log_probability(self, theta, t, rv, rv_err, p, a, b, t0, period):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta, t, rv, rv_err, p, a, b, t0, period)

    # ------------------------------------------------------------------
    def quick_fit(self, t, rv, rv_err, p, a, b, t0, period):
        """
        Fast scipy curve_fit point estimate to seed MCMC walkers.

        Returns
        -------
        ndarray [v_sin_i, lambda]
        """
        def model_fn(t_, v_sin_i, lam):
            return self.ohta_rm_profile(t_, t0, period, v_sin_i, lam, p, a, b)

        try:
            popt, _ = curve_fit(model_fn, t, rv, p0=[5.0, 0.0],
                                sigma=rv_err,
                                bounds=([0.0, -np.pi], [200.0, np.pi]),
                                maxfev=5000)
            return popt
        except Exception:
            return np.array([5.0, 0.0])

    # ------------------------------------------------------------------
    def run_mcmc(self, t, rv, rv_err, p_ratio, a_ratio, b, t0, period):
        """
        Run full MCMC posterior sampling for v_sin_i and lambda.

        Parameters
        ----------
        t, rv, rv_err : array  Radial velocity time series during transit
        p_ratio       : float  Rp/R_* (from photometric MCMC)
        a_ratio       : float  a/R_* (from photometric MCMC)
        b             : float  Impact parameter (from photometric MCMC)
        t0, period    : float  Transit epoch and orbital period

        Returns
        -------
        results : dict   Median and 1-sigma for v_sin_i and lambda
        samples : ndarray  Flat chain for corner plots
        """
        print("[RMSolverMCMC] Pre-fitting with curve_fit...")
        initial = self.quick_fit(t, rv, rv_err, p_ratio, a_ratio, b, t0, period)
        print(f"  Seed: v_sin_i={initial[0]:.2f} km/s  lambda={np.degrees(initial[1]):.1f} deg")

        ndim = 2
        pos  = initial + 1e-3 * np.random.randn(self.nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, self._log_probability,
            args=(t, rv, rv_err, p_ratio, a_ratio, b, t0, period)
        )
        print(f"[RMSolverMCMC] Running {self.nsteps} iterations...")
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        flat = sampler.get_chain(discard=self.nburn, thin=self.nthin, flat=True)

        results = {}
        for i, name in enumerate(["v_sin_i", "lambda_rad"]):
            lo, med, hi = np.percentile(flat[:, i], [16, 50, 84])
            results[name] = {
                "median":    float(med),
                "lower_err": float(med - lo),
                "upper_err": float(hi - med),
                "samples":   flat[:, i],
            }

        results["lambda_deg"] = {
            "median":    float(np.degrees(results["lambda_rad"]["median"])),
            "lower_err": float(np.degrees(results["lambda_rad"]["lower_err"])),
            "upper_err": float(np.degrees(results["lambda_rad"]["upper_err"])),
        }

        amp = self.theoretical_amplitude(
            results["v_sin_i"]["median"], p_ratio, b
        )
        results["theoretical_amplitude_km_s"] = amp

        lam_deg = results["lambda_deg"]["median"]
        if abs(lam_deg) < 20:
            results["tidal_state"] = "Aligned — consistent with tidal synchronization"
        elif abs(lam_deg) < 60:
            results["tidal_state"] = "Moderate misalignment — possible atmospheric drag"
        else:
            results["tidal_state"] = "High misalignment — retrograde/polar orbit"

        print(f"[RMSolverMCMC] Done.")
        print(f"  v sin i = {results['v_sin_i']['median']:.2f} "
              f"+{results['v_sin_i']['upper_err']:.2f}/-{results['v_sin_i']['lower_err']:.2f} km/s")
        print(f"  lambda  = {results['lambda_deg']['median']:.1f} "
              f"+{results['lambda_deg']['upper_err']:.1f}/-{results['lambda_deg']['lower_err']:.1f} deg")
        print(f"  State:    {results['tidal_state']}")
        return results, flat


# ─────────────────────────────────────────────────────────────────────────────
# Helper: estimate v sin i from stellar Teff (Kraft 1967 + modern calibrations)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_v_sin_i(Teff_K: float) -> float:
    """
    Empirical v sin i estimate from stellar Teff.

    Based on the Kraft (1967) rotation-temperature relation:
    - Stars hotter than ~6250 K lack a deep convection zone -> fast rotators
    - Stars cooler than ~6250 K have convective envelopes -> magnetic braking
      -> slow rotation (v sin i ~ 1-6 km/s), approaching zero for old M dwarfs

    References: Kraft (1967, ApJ, 150, 551); Reinhold et al. (2013, A&A)
    """
    if Teff_K < 4000:
        return 1.0    # M dwarf: very slow
    elif Teff_K < 4500:
        return 1.5    # late K
    elif Teff_K < 5200:
        return 2.5    # K dwarf
    elif Teff_K < 5800:
        return 3.5    # G dwarf (solar-like)
    elif Teff_K < 6200:
        return 5.5    # warm G / early G
    elif Teff_K < 6700:
        return 15.0   # F star (past Kraft break)
    else:
        return 35.0   # hot F / A star


def estimate_geometric_params(R_planet_Rearth: float, M_star_Msun: float,
                               a_AU: float) -> dict:
    """
    Estimate Rp/R*, a/R*, and b from stellar mass, planet radius, semi-major axis.

    Stellar radius from Demircan & Kahraman (1991) mass-radius relation:
        R_* / R_sun = (M_* / M_sun)^0.80   (main sequence approximation)

    Returns
    -------
    dict  p_ratio (Rp/R*), a_ratio (a/R*), b (default 0.3)
    """
    R_SUN_IN_REARTH = 109.076   # 1 R_sun in Earth radii
    AU_IN_RSUN      = 215.032   # 1 AU in solar radii

    R_star_Rsun = M_star_Msun ** 0.80          # stellar radius estimate
    R_star_Rearth = R_star_Rsun * R_SUN_IN_REARTH

    p_ratio = R_planet_Rearth / R_star_Rearth  # Rp/R*
    a_ratio = (a_AU * AU_IN_RSUN) / R_star_Rsun  # a/R* in same solar units

    return {
        "p_ratio":      float(np.clip(p_ratio, 0.001, 0.25)),
        "a_ratio":      float(max(a_ratio, 2.0)),
        "b":            0.3,                   # default impact parameter
        "R_star_Rsun":  float(R_star_Rsun),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extended 4-parameter RM MCMC
# ─────────────────────────────────────────────────────────────────────────────

class RMSolverMCMCExtended(RMSolverMCMC):
    """
    Extended Rossiter-McLaughlin MCMC with 4 free parameters:

      theta = [v_sin_i, lambda, b, delta_rv0]

    Parameters
    ----------
    v_sin_i   : projected stellar equatorial velocity (km/s)
    lambda    : spin-orbit angle (radians)
    b         : impact parameter  [0, 1)   — previously fixed from photometry
    delta_rv0 : RV baseline offset (km/s)  — handles instrumental drift / stellar activity

    Why these 4?
    - v_sin_i and lambda are the primary science parameters
    - b affects the RM amplitude AND shape; fixing it from photometry assumes
      no systematic error in the transit light curve — freeing it here reveals
      the b-lambda degeneracy (documented in the corner plot)
    - delta_rv0 absorbs any RV zero-point offset or stellar activity residual
    - Rp/Rs and a/Rs are NOT freed: they are so well constrained by photometry
      that adding them creates severe degeneracies with v_sin_i without
      adding scientific value

    Priors
    ------
    v_sin_i  : Uniform [0, 200 km/s]
    lambda   : Uniform [-pi, pi]
    b        : Uniform [0, 0.99]
    delta_rv0: Gaussian(0, sigma_rv0)  — weakly informative
    """

    def __init__(self, nwalkers=32, nsteps=2000, nburn=400, nthin=10,
                 sigma_rv0=0.5):
        super().__init__(nwalkers=nwalkers, nsteps=nsteps,
                         nburn=nburn, nthin=nthin)
        self.sigma_rv0 = sigma_rv0   # prior width on RV offset (km/s)

    # ------------------------------------------------------------------
    def _log_prior_ext(self, theta):
        """Priors on [v_sin_i, lambda, b, delta_rv0]."""
        v_si, lam, b, drv = theta
        if not (0.0 < v_si < 200.0):
            return -np.inf
        if not (-np.pi < lam < np.pi):
            return -np.inf
        if not (0.0 <= b < 0.99):
            return -np.inf
        # Gaussian prior on RV offset
        lp_drv = -0.5 * (drv / self.sigma_rv0)**2
        return lp_drv

    def _log_likelihood_ext(self, theta, t, rv, rv_err, p, a, t0, period):
        v_si, lam, b, drv = theta
        model  = self.ohta_rm_profile(t, t0, period, v_si, lam, p, a, b) + drv
        sigma2 = rv_err**2
        return -0.5 * np.sum((rv - model)**2 / sigma2 + np.log(sigma2))

    def _log_prob_ext(self, theta, t, rv, rv_err, p, a, t0, period):
        lp = self._log_prior_ext(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood_ext(theta, t, rv, rv_err, p, a, t0, period)

    # ------------------------------------------------------------------
    def quick_fit_ext(self, t, rv, rv_err, p, a, b_init, t0, period):
        """Seed the extended MCMC with a curve_fit estimate."""
        from scipy.optimize import curve_fit as _cf

        def model_fn(t_, v_si, lam, b_, drv):
            return self.ohta_rm_profile(t_, t0, period, v_si, lam, p, a, b_) + drv

        try:
            popt, _ = _cf(model_fn, t, rv,
                          p0=[5.0, 0.0, b_init, 0.0],
                          sigma=rv_err,
                          bounds=([0, -np.pi, 0, -2], [200, np.pi, 0.99, 2]),
                          maxfev=8000)
            return popt
        except Exception:
            return np.array([5.0, 0.0, b_init, 0.0])

    # ------------------------------------------------------------------
    def run_mcmc_extended(self, t, rv, rv_err, p_ratio, a_ratio, b_init,
                           t0, period):
        """
        Run 4-parameter MCMC: v_sin_i, lambda, b, delta_rv0.

        Returns
        -------
        results : dict  Posteriors for all 4 parameters
        samples : ndarray  Shape (N, 4) flat chain
        """
        print("[RMSolverMCMCExtended] Pre-fitting (4-param)...")
        init = self.quick_fit_ext(t, rv, rv_err, p_ratio, a_ratio,
                                   b_init, t0, period)
        print(f"  Seed: v={init[0]:.2f} lam={np.degrees(init[1]):.1f} "
              f"b={init[2]:.2f} drv={init[3]:.3f}")

        ndim = 4
        pos  = init + 1e-3 * np.random.randn(self.nwalkers, ndim)
        # Clip b walkers to [0, 0.98]
        pos[:, 2] = np.clip(pos[:, 2], 0.0, 0.98)

        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, self._log_prob_ext,
            args=(t, rv, rv_err, p_ratio, a_ratio, t0, period)
        )
        print(f"[RMSolverMCMCExtended] Running {self.nsteps} steps...")
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        flat = sampler.get_chain(discard=self.nburn, thin=self.nthin, flat=True)

        param_names = ["v_sin_i", "lambda_rad", "b", "delta_rv0"]
        results = {}
        for i, name in enumerate(param_names):
            lo, med, hi = np.percentile(flat[:, i], [16, 50, 84])
            results[name] = {
                "median":    float(med),
                "lower_err": float(med - lo),
                "upper_err": float(hi - med),
            }

        results["lambda_deg"] = {
            "median":    float(np.degrees(results["lambda_rad"]["median"])),
            "lower_err": float(np.degrees(results["lambda_rad"]["lower_err"])),
            "upper_err": float(np.degrees(results["lambda_rad"]["upper_err"])),
        }

        lam_d = results["lambda_deg"]["median"]
        if abs(lam_d) < 20:
            results["tidal_state"] = "Aligned — consistent with tidal synchronization"
        elif abs(lam_d) < 60:
            results["tidal_state"] = "Moderate misalignment"
        else:
            results["tidal_state"] = "High misalignment — retrograde/polar orbit"

        amp = self.theoretical_amplitude(
            results["v_sin_i"]["median"], p_ratio,
            results["b"]["median"]
        )
        results["theoretical_amplitude_km_s"] = amp

        print(f"[RMSolverMCMCExtended] Done.")
        print(f"  v sin i = {results['v_sin_i']['median']:.2f} km/s")
        print(f"  lambda  = {results['lambda_deg']['median']:.1f} deg")
        print(f"  b       = {results['b']['median']:.3f}")
        print(f"  drv0    = {results['delta_rv0']['median']:.4f} km/s")
        return results, flat

