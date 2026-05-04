# Constraining Exoplanetary Spin States
## A Complete Physics Guide to the Tidal Locking Pipeline

---

## 1. What Is Tidal Locking?

### 1.1 The Basic Idea

Every massive body in space raises a **tidal bulge** on any nearby body. The Moon raises tides
on Earth (ocean and solid-body). Earth raises a much larger tidal bulge on the Moon.

The key physics: **if the planet spins faster than it orbits, the tidal bulge runs ahead of
the star-planet line**. The star's gravity then pulls this leading bulge backward — applying a
torque that slows the planet's spin. Over geological timescales, the spin rate decreases until
it matches the orbital period. The planet is then *tidally locked* (or *synchronously rotating*):
one face permanently points toward the star.

The Moon is tidally locked to Earth — that is why we always see the same face.

### 1.2 Why Close-In Planets Lock First

The tidal force scales as **1/r³** (tidal forces are differential gravity). The torque scales
even more steeply. The synchronization timescale goes as **a⁶** (sixth power of orbital
distance). This makes inner planets lock on timescales of millions of years, while outer planets
like Jupiter take longer than the age of the Universe.

---

## 2. The Gladman (1996) Synchronization Timescale

### 2.1 Derivation from First Principles

Start from Goldreich & Soter (1966). The tidal potential raised on a planet of radius R_p by
a star of mass M_* at distance r can be written to second order (quadrupole approximation):

```
U_tidal = -(G M_* / r^3) * k2 * R_p^5 / r^3  * (angular terms)
```

where `k2` is the **Love number of degree 2**: how much the planet deforms relative to a
fluid body (k2=1.5 for fluid, ~0.3 for rocky Earth, ~0.1–0.5 for ice giants).

The lag angle between the raised bulge and the star-planet line is parameterized by the
**tidal quality factor Q**. A planet with Q=100 dissipates tidal energy much more efficiently
than one with Q=10⁶ (Q is inversely proportional to dissipation). Rocky planets: Q ~ 10–500.
Giant planets: Q ~ 10⁴–10⁶.

The tidal torque on the planet is:

```
T = -(3/2) * G * M_*^2 * k2 * R_p^5 / (Q * r^6) * (omega - n)
```

where `omega` is the planet spin rate and `n = 2*pi/P_orb` is the orbital mean motion.

For `omega > n` (spinning faster than orbital motion), T < 0 (braking torque).

Integrating dL/dt = T, where L = xi * M_p * R_p^2 * omega is the rotational angular
momentum (xi = 0.33 for rocky, 0.25 for gas/ice giant — the moment of inertia factor):

```
tau_sync = (2*pi*xi/9) * omega_0 * M_p * Q * a^6
           / (G * k2 * M_*^2 * R_p^3)
```

This is **Gladman et al. (1996), Eq. 5**.

### 2.2 Physical Interpretation of Each Term

| Term | Effect on tau_sync | Physical meaning |
|---|---|---|
| `a^6` | Very strong increase | Tidal force drops steeply with distance |
| `Q` | Linear increase | High-Q planets dissipate less → lock slower |
| `M_p` | Linear increase | More massive = more angular momentum to remove |
| `R_p^-3` | Strong decrease | Larger planet → stronger tidal deformation |
| `M_*^-2` | Decrease | More massive star → stronger tidal torque |
| `k2^-1` | Increase | Less deformable planet (low k2) → locks slower |
| `omega_0` | Linear increase | Faster initial spin → more angular momentum to remove |

### 2.3 The a^6 Dominance

The **a^6 dependence is the dominant factor**. Going from 0.1 AU to 1 AU increases tau_sync
by 10^6. This is why:
- HD 219134 b at 0.038 AU → locked in < 1 Myr
- HD 10700 f at 1.33 AU → NOT locked even after 5.8 Gyr

Our empirical fit found alpha = -6.87 ± 0.20 for log(age/tau) vs log(a). Theory predicts
-6. The slight steepening is consistent with the spread in stellar masses and Q values across
the sample.

### 2.4 The Q-k2 Uncertainty Problem

Q spans 3 orders of magnitude for rocky planets (10 to 10,000). This means tau_sync is
uncertain by the same factor. **No amount of better observations will pin down Q** — it
depends on the unmeasured interior structure of the planet.

This is why we show TWO error bars on every plot:
1. **Measurement errors** (a, M_*, R_p, M_p) — about 64% fractional
2. **Q-model errors** (1 dex) — a factor of 10 each way

The Q errors dominate. But they affect all planets equally, so they shift the whole
population, not the relative order. The empirical ordering (which planets are locked
relative to each other) is robust.

---

## 3. Pseudo-Synchronous Rotation (Hut 1981)

### 3.1 Why Eccentric Orbits Don't Lock to 1:1

For a circular orbit, the equilibrium spin rate is exactly the orbital mean motion:
omega_eq = n = 2*pi/P_orb

For an **eccentric orbit**, the orbital angular velocity varies:
- At pericenter (closest approach): angular velocity is MAXIMUM
- At apocenter (farthest point): angular velocity is MINIMUM

The tidal torque is strongest at pericenter (because force ∝ 1/r^6). So the planet "wants"
to spin at the pericenter angular velocity. The equilibrium is a **pseudo-synchronous** rate
that is faster than the mean orbital motion:

```
omega_ps/n = [1 + (15/2)e^2 + (45/8)e^4 + (5/16)e^6]
             / [(1-e^2)^(3/2) * (1 + 3e^2 + (3/8)e^4)]
```

This is **Hut (1981), Eq. 42**.

### 3.2 Practical Implications

For Earth's eccentricity (e=0.017): P_ps/P_orb ≈ 1.0002 (essentially locked).
For HD 3651 b (e=0.645): P_ps/P_orb ≈ 0.42 (rotates 2.4x faster than orbital period!).

Mercury is a special case — it settled into a **3:2 spin-orbit resonance** rather than 1:1
or pseudo-sync. This is discussed next.

---

## 4. Spin-Orbit Resonances

### 4.1 Why Mercury Chose 3:2

During tidal despinning, a planet passes through commensurability ratios (omega/n = p/q where
p,q are small integers). For a non-zero eccentricity, the gravitational torque has a
time-varying component that can "trap" the spin at these resonances before reaching 1:1.

The capture probability into a resonance increases with eccentricity. Mercury has e=0.206 and
was captured into 3:2 (spins 3 times per 2 orbits) before reaching synchronous rotation.

Our code tests resonances 3:2, 5:2, 2:1, 5:3 using:

```
|P_rot / P_orb - p/q| < tolerance
```

### 4.2 For HWO Targets

Most HWO rocky planets have low eccentricities (e < 0.1 from radial velocity fitting), so
most are either in 1:1 lock or freely rotating. The few with moderate eccentricities
(HD 3651 b, e=0.645) are likely in pseudo-synchronous rotation.

---

## 5. The Rossiter-McLaughlin Effect

### 5.1 What It Is

When a planet transits its star, it blocks part of the stellar disk. The star is rotating,
so one limb is approaching us (blueshifted) and the other is receding (redshifted). When
the planet blocks the approaching limb (ingress for a prograde orbit), the integrated stellar
light becomes slightly more redshifted. When it covers the receding limb (egress), it becomes
more blueshifted.

This creates an **RV anomaly** during transit: first a positive (red) spike, then a negative
(blue) spike for a prograde aligned orbit (lambda = 0).

### 5.2 The Ohta-Taruya-Suto (2005) Model

The RM anomaly at time t depends on the position of the planet center on the stellar disk.
In the kinematic approximation (Ohta et al. 2005):

```
x_c(t) = a/R_* * sin(2*pi*(t-t0)/P)       [planet x-position in sky plane]
y_c    = b                                  [impact parameter, fixed]
x_rot  = x_c * cos(lambda) - y_c * sin(lambda)   [rotated by spin-orbit angle]

DeltaRV(t) = v_sin_i * x_rot * (Rp/R*)^2 * edge_damp(t)
```

where `edge_damp` smooths the ingress/egress when only part of the planet overlaps the disk.

The amplitude is (Cristo et al. 2024):

```
Delta_v_max = v_sin_i * (Rp/R_*)^2 * sqrt(1 - b^2)
```

### 5.3 Physical Parameters

| Parameter | Physical meaning | How we measure it |
|---|---|---|
| `v sin i` | Projected stellar equatorial rotation speed | From broadening of stellar absorption lines |
| `lambda` | Spin-orbit angle (sky projection) | From RM anomaly shape |
| `b` | Impact parameter (0=central transit) | From photometric transit shape |
| `Rp/R_*` | Planet-to-star radius ratio | From transit depth |
| `a/R_*` | Scaled orbital separation | From transit duration |

### 5.4 What lambda Tells Us

- lambda ≈ 0° → prograde, aligned orbit. Consistent with disk migration and tidal locking.
- lambda ≈ 180° → retrograde orbit. Planet probably got its orbit flipped by Kozai-Lidov cycles.
- lambda > 20° → some misalignment. Could be primordial disk warp, planet-planet scattering,
  or stellar magnetic activity affecting the measured angle.
- For tidally locked planets, lambda should be near 0° because tidal damping also damps
  orbital inclination over long timescales.

### 5.5 The 4-Parameter Extended MCMC

The basic 2-param MCMC (v_sin_i, lambda) treats b as known from photometry. Our extended fit
adds:

- **b (impact parameter)**: Freeing this reveals the b-lambda degeneracy. A transit closer to
  the stellar limb (high b) creates an asymmetric RM profile that looks like a misaligned orbit.
  The corner plot will show a banana-shaped b-lambda correlation — this is expected.

- **delta_rv0 (RV offset)**: Absorbs any instrumental zero-point drift or stellar activity
  residuals. If delta_rv0 is large (> 0.1 km/s), it indicates systematics not captured by
  the noise model.

---

## 6. Atmospheric Super-Rotation and Doppler Signatures

### 6.1 Why Atmospheres Super-Rotate

Even on a tidally locked planet, the atmosphere is NOT perfectly co-rotating. The day side
is hot, the night side is cold. Atmospheric dynamics (Coriolis + pressure gradients) drive
a net eastward (prograde) flow — a **super-rotating equatorial jet** — at the speed of sound
scale (~1 km/s for hot Jupiters, ~0.1-0.3 km/s for rocky super-Earths).

For a **non-tidally-locked planet**, this super-rotation adds on top of the planet's rotation.
The morning limb (ingress) is moving away from the observer: **redshift**.
The evening limb (egress) is moving toward the observer: **blueshift**.
The net shift is: **Delta_v_atm = v_ingress - v_egress** (positive for a super-rotating jet).

### 6.2 The Key Diagnostic

| Planet state | v_ingress | v_egress | Delta_v_atm | Signature |
|---|---|---|---|---|
| Tidally locked | small, symmetric | small, symmetric | ~0 km/s | Lines overlap |
| Free rotator | +1.8 km/s (red) | -2.2 km/s (blue) | +4.0 km/s | Lines shifted |

This asymmetry was first detected by Snellen et al. (2010) for HD 209458b using CO lines
in the near-infrared (CRIRES spectrograph). The shift was ~2 km/s, confirming a
super-rotating wind in that planet's atmosphere.

### 6.3 How We Detect It

Using high-resolution transmission spectroscopy (R > 100,000), we can resolve individual
absorption lines of CO, H2O, Na, K. The ingress spectrum is taken during first contact,
the egress spectrum during last contact. The Doppler shift between these two spectra gives
Delta_v_atm directly.

Current instruments: CRIRES+ (VLT), ESPRESSO (VLT), NIRPS, future ELT/HIRES.

### 6.4 Our Forward Model

For batch processing, we estimate Delta_v for each planet using the Showman & Guillot (2002)
scaling:

```
Delta_v ~ Delta_v_fid * (v_orb / v_orb_fid) * sqrt(R_p / R_fid)
```

calibrated to HD 209458b (fiducial: v_orb=152 km/s, R_p=15 R_Earth, Delta_v=4 km/s).

---

## 7. Doppler Decomposition: Axial vs Super-Rotation

### 7.1 Two Sources of Doppler Shift

In transmission spectroscopy, we measure the total Doppler shift difference between the morning limb (ingress) and evening limb (egress):

`Delta_v_total = v_ingress - v_egress`

This total shift has two distinct physical sources:
1. **Axial Rotation (`Delta_v_rot`)**: The planet's own spin. For a rigid rotator, `v_eq = 2*pi*R_p / P_rot`. Both limbs contribute equally, so `Delta_v_rot = 2 * v_eq`. If the planet is tidally locked (`P_rot = P_orb`), this contribution is very small because orbital periods are days-to-months. If it's a free rotator, this can be large.
2. **Super-Rotation Jet (`Delta_v_super`)**: A fast eastward atmospheric jet driven by day-night temperature differences. This is present even on locked planets.

### 7.2 The Decomposition Strategy

Our pipeline decomposes the total signal to isolate the planet's rotation from the atmospheric jet:
```
Delta_v_rot_excess = 2 * (v_eq_free - v_eq_locked)
```
This isolates the *excess* rotation above the tidally locked baseline. 

The super-rotation is estimated via Showman & Guillot (2002) scaling with equilibrium temperature and planet radius.

### 7.3 Why This Matters

This decomposition is the **novel diagnostic of our pipeline**. By splitting `Delta_v_total`, we can determine *which* physical effect dominates the observable signal. If `Delta_v_rot_excess > Delta_v_super`, we can directly measure the planet's spin state from the transmission spectrum, independent of the tidal synchronization theory. 

---

## 8. Observational Detection Feasibility

### 8.1 Photon Budgets and SNR

To detect a Doppler shift `Delta_v` at 5-sigma significance, we need high spectral resolution (`R = 100,000`) and high Signal-to-Noise Ratio (SNR).

The velocity precision per transit is given by:
```
sigma_v = (c / R) / (sqrt(N_lines) * SNR)
```
where `N_lines` is the number of molecular absorption lines in the bandpass (e.g., ~15 for rocky planets with H2O/CO, ~60 for gas giants) and `SNR = sqrt(N_photons)`.

### 8.2 Required Transits

The precision improves with the square root of the number of transits observed:
```
N_transits = (5 * sigma_v / Delta_v_total)^2
```
Our pipeline computes this for every HWO target for major future instruments (HWO 6m, ELT-HIRES 39m, VLT/CRIRES+ 8m). This directly answers the strategic question: *Which planets should we point ELT at first to test the tidal locking theory?*

---

## 9. Error Propagation: Two Independent Sources

### 9.1 Measurement Uncertainties

From the Gladman formula tau ∝ M_p * a^6 / (M_*^2 * R_p^3):

```
(delta_tau/tau)^2 = (delta_Mp/Mp)^2 + (6*delta_a/a)^2
                  + (2*delta_Mstar/Mstar)^2 + (3*delta_Rp/Rp)^2
```

For typical HWO targets:
- delta_a/a = 0.5%  → contribution: (6 × 0.005)^2 = 0.09%  (negligible)
- delta_Mstar/Mstar = 5% → contribution: (2×0.05)^2 = 1%
- delta_Rp/Rp = 20% → contribution: (3×0.2)^2 = 36%  (DOMINANT)
- delta_Mp/Mp = 20% → contribution: 4%

Total measurement fractional error: **~64%**

The R_p uncertainty dominates because tau ∝ R_p^{-3}, and R_p is estimated from mass
via the Chen & Kipping (2017) mass-radius relation, which has ~20% scatter.

### 9.2 Q-Model Uncertainties

The tidal quality factor Q is completely unmeasured for exoplanets. Earth's Q ≈ 12.
Mars's Q ≈ 80. The range for rocky exoplanets is Q = 10 to 10,000 (3 orders of magnitude).

Since tau ∝ Q, this gives a **factor of 1000 uncertainty on tau_sync** from Q alone.
We represent this as ±1 dex in log space, i.e., tau could be 10× larger or 10× smaller
than our fiducial estimate.

### 9.3 How to Use Both on Plots

Both error sets are shown:
- **Error bars**: from measurement uncertainties (the 64% calculation above)
- **Shaded band (dashed)**: from Q uncertainty (±1 dex)

A planet is considered "confidently locked" only if even the upper Q bound gives age/tau > 1.
A planet is "confidently free" only if even the lower Q bound gives age/tau < 1.

---

## 10. Empirical Correlation Analysis

### 10.1 The Power-Law Model

We fit:
```
log10(age / tau_sync) = C + alpha * log10(a) + beta * log10(R_p)
```

using ordinary least squares on the batch results. This is a 2D plane fit in log-log space.

### 10.2 What the 1D Fits Show

**1D fit: log(age/tau) vs log(a)**
- alpha = -6.87 ± 0.20
- Theory (Gladman): alpha = -6
- Pearson r = -0.97 → almost perfect anti-correlation

**1D fit: log(age/tau) vs log(R)**
- beta = -7.72 ± 0.99
- Theory (for fixed a): beta should be +3 (larger R_p → smaller tau → larger age/tau)
- But we get NEGATIVE beta! Why?

**The confounding variable explanation:**
The 1D beta is contaminated because larger planets (gas giants, R ~ 10-15 R_Earth) are
systematically at LARGER orbital distances (a > 2 AU) in our sample. Distance dominates
via the a^6 factor, making age/tau small for large-R planets. After controlling for a
in the 2D fit, beta becomes -0.22 (near zero), which means R_p has almost no residual
predictive power once a is known. This is physically expected — the a^6 dependence
completely dominates over the R_p^3 dependence for this population.

### 10.3 Confidence Bands

The 1-sigma confidence band on the power-law fit is computed from the covariance matrix of
the linear regression coefficients:

```
sigma_pred(x)^2 = sigma_C^2 + x^2 * sigma_alpha^2 + 2*x*cov(C,alpha)
```

This gives a hyperbolic band that is narrowest at the center of the data and widens at the
edges. The band represents uncertainty in the FIT LINE, not scatter in the data.

---

## 11. Reading Each Plot

### Plot 1: `tidal_evolution.png`
**What it shows**: tau_sync vs orbital distance a (log-log) for rocky, ice, gas planets.
**The horizontal line** at the system age separates locked (below) from free (above).
**Why log-log**: tau spans 15 orders of magnitude across 0.01–5 AU.

### Plot 2: `rm_anomaly.png`
**What it shows**: RV time-series during transit + best-fit OTS model.
**Red points**: ingress (planet on approaching stellar limb → redshift).
**Blue points**: egress (planet on receding stellar limb → blueshift).
**The shape** of the curve tells you lambda: symmetric = aligned, skewed = misaligned.
**Error bars**: heteroscedastic noise (larger near ingress/egress where stellar limb-darkening
changes rapidly).

### Plot 3: `rm_corner.png`
**What it shows**: MCMC posterior distributions for v_sin_i and lambda (or 4 params).
**Diagonal panels**: 1D marginal posteriors for each parameter.
**Off-diagonal panels**: 2D joint posteriors (contours) showing parameter correlations.
**Key feature** in 4-param corner: banana-shaped b-lambda contour = degeneracy.

### Plot 4: `synthetic_unlocked_rv.png` and `synthetic_locked_rv.png`
**What they show**: simulated RV anomaly for a free rotator vs a locked planet.
**The key difference**: the locked planet has SYMMETRIC RM and tiny (< 0.2 km/s) atmospheric
shift. The unlocked planet has ASYMMETRIC atmospheric shift of ~4 km/s.

### Plot 5: `locked_vs_unlocked_comparison.png`
**What it shows**: 4-panel comparison (RV anomaly + transmission spectra for locked and unlocked).
**Top left**: locked RV — symmetric RM, no net atmospheric shift.
**Top right**: unlocked RV — RM + clear ingress/egress asymmetry.
**Bottom left**: locked spectra — ingress and egress lines OVERLAP.
**Bottom right**: unlocked spectra — lines SEPARATED by ~0.02 nm (= ~4 km/s at 589 nm).

### Plot 6: `batch_locking_map.png`
**What it shows**: Each planet in (a, R_p) space, colored by locking state.
**Filled circles**: locked. Open circles: free.
**The boundary line** (when RUN_CORRELATION=True): empirical locking threshold.
**Error bars**: measurement uncertainty on log10(age/tau) — vertical bars on the right axis.

### Plot 7: `empirical_fit.png`
**What it shows**: power-law regression of log(age/tau) vs log(a) and log(R).
**Gray band**: 1-sigma confidence interval on the fit.
**Dashed theory line**: theoretical expectation from Gladman (1996).
**Rocky planets**: filled, colored by locking state.

### Plot 8: `wind_signature_population.png`
**What it shows**: predicted atmospheric Delta_v for each planet.
**Locked planets**: Delta_v = 0 (co-rotating atmosphere).
**Free rotators**: Delta_v > 0, scaled by orbital velocity and planet size.
**This is an observational prediction** for future CRIRES+/ELT follow-up.

### Plot 9: `sensitivity_dashboard.png`
**What it shows**: sensitivity of locking conclusion to Q and k2 uncertainties.
**Color grid**: locked fraction over (Q, k2) parameter space.
**The point**: even with order-of-magnitude Q uncertainty, the locked/free classification
is robust for planets deep in the locked or free regime.

### Plot 10: `rv_anomaly_gallery.png`
**What it shows**: RV anomaly for all 70 HWO planets in a multi-panel grid.
**Each panel**: a single planet's synthetic RM anomaly + MCMC best fit.
**Purpose**: visual quality check that the MCMC converged for all planets.

### Plot 11: `doppler_decomposition.png`
**What it shows**: Stacked bar chart separating `Delta_v_rot_excess` (axial rotation) and `Delta_v_super` (atmospheric jet) for each planet.
**Purpose**: Shows which planets are dominated by solid-body rotation vs atmospheric dynamics.

### Plot 12: `detection_feasibility.png`
**What it shows**: 2-panel figure. Left: Detection landscape (Signal vs Noise). Right: Top 20 priority planets ranked by fewest transits needed.
**Purpose**: An actionable observing schedule for HWO and ELT.

---

## 12. MCMC: What It Is and Why We Use It

### 12.1 Why Not Just curve_fit?

`scipy.optimize.curve_fit` gives point estimates (best-fit values) and a covariance matrix
(formal errors assuming linear error propagation and Gaussian noise). It works well when:
- The noise is Gaussian and homoscedastic
- The model is approximately linear near the minimum
- There are no parameter degeneracies

For the RM problem:
- Our noise IS heteroscedastic (larger errors at ingress/egress)
- v_sin_i and lambda ARE degenerate (both affect amplitude)
- b and lambda ARE degenerate (both affect shape)

MCMC gives the **full posterior distribution**, correctly propagating all degeneracies.

### 12.2 How emcee Works (MCMC Ensemble Sampler)

We use `emcee` (Foreman-Mackey et al. 2013) — the "affine-invariant ensemble sampler":

1. Initialize N_walkers (64) starting points near the curve_fit solution.
2. Each walker proposes a new position based on the positions of OTHER walkers.
3. Accept/reject based on the Metropolis-Hastings criterion: accept if the new point
   has higher probability, accept probabilistically if it has lower probability.
4. After N_steps (10,000), discard the "burn-in" phase (first 3,000 steps where chains haven't
   converged), thin by factor N_thin (15) to reduce autocorrelation.
5. The remaining samples directly represent the posterior distribution.

### 12.3 Reading Convergence

Good MCMC chains look like:
- Acceptance fraction should be 0.2–0.5
- Autocorrelation length should be << N_steps / N_thin

The corner plot is the best visual convergence check: smooth, Gaussian-like 1D histograms
and elliptical 2D contours indicate good convergence.

---

## 13. The Publication Pathway

### What Result Is Potentially Publishable

1. **Empirical scaling law**: alpha = -6.87 ± 0.20 vs Gladman theory -6. A paper comparing
   empirical tidal locking rates across the HWO target list to theoretical predictions,
   discussing Q-factor constraints, would be suitable for A&A or AJ.

2. **The confounding beta result**: the sign flip in the 1D beta is a pedagogically important
   result about selection effects in exoplanet population studies.

3. **Rotational Decomposition & Observability**: The decomposition of atmospheric signatures into axial and super-rotational components, and the resulting ELT/HWO feasibility priority list, is a novel and directly actionable result.

### What You Would Need for a Real Paper

- **Real RM data** from HARPS/ESPRESSO for at least the closest HWO targets
- **Stellar v sin i** measurements from spectroscopic catalogs (GALAH, APOGEE, HARPS archive)
- **Better Q constraints** from tidal dissipation studies of solar system bodies extrapolated
  to exoplanets
- A comparison of your empirical boundary with independent tidal models (Leconte et al. 2010,
  Heller et al. 2011)

---

## 14. Quick Reference — Physical Constants Used

| Symbol | Value | Meaning |
|---|---|---|
| G | 6.674e-11 N m^2 kg^-2 | Gravitational constant |
| M_sun | 1.989e30 kg | Solar mass |
| R_Earth | 6.371e6 m | Earth radius |
| M_Earth | 5.972e24 kg | Earth mass |
| AU | 1.496e11 m | Astronomical unit |
| omega_Earth | 7.29e-5 rad/s | Earth rotation rate (initial spin for rocky planets) |
| Year | 3.156e7 s | Julian year |
| xi_rocky | 0.33 | Moment of inertia factor, rocky planets |
| xi_gas | 0.25 | Moment of inertia factor, gas/ice giants |
| Q_rocky | 100 (fiducial) | Tidal quality factor, rocky (range: 10–10000) |
| k2_rocky | 0.299 (Earth) | Love number degree 2, rocky |
| Q_gas | 10000 (fiducial) | Tidal quality factor, gas giants (range: 10^4–10^6) |
| k2_gas | 0.5 | Love number degree 2, gas giants |

---

*Guide written for the HWO Population Tidal Analysis pipeline — Module 2.*
*All derivations follow Gladman et al. (1996), Hut (1981), Ohta et al. (2005), Snellen et al. (2010).*
