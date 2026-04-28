"""
rotation_analyser.py  (Module 2 — Tidal Locking)
==================================================
Derives the rotation state of exoplanets from orbital parameters.

Covers:
  1. Synchronous (tidally locked) rotation check — tau_sync vs system age
  2. Pseudo-synchronous rotation for eccentric orbits (Hut 1981)
  3. Spin-orbit resonance detection (1:1, 3:2, 2:1, etc.)
  4. Q-k2 sensitivity grid (Goldreich & Soter 1966)

References:
  - Gladman et al. 1996, Icarus, 122, 166         (sync timescale)
  - Goldreich & Soter 1966, Icarus, 5, 375        (tidal Q)
  - Hut 1981, A&A, 99, 126                        (pseudo-synchronous rotation)
  - Murray & Dermott 1999, Solar System Dynamics   (resonance theory)
  - Chen & Kipping 2017, ApJ, 834, 17             (mass-radius relations)
"""

import numpy as np

# Physical constants (SI)
G      = 6.674e-11
M_SUN  = 1.989e30
M_EARTH= 5.972e24
R_EARTH= 6.371e6
AU     = 1.496e11
DAY_S  = 86400.0
YEAR_S = 365.25 * DAY_S
OMEGA_EARTH = 7.2921e-5   # rad/s

# Default tidal parameters by planet type
TIDAL_Q = {
    "rocky":     100,
    "ice_giant": 1000,
    "gas_giant": 1e5,
}
LOVE_K2 = {
    "rocky":     0.299,
    "ice_giant": 0.3,
    "gas_giant": 0.37,
}


class RotationAnalyser:
    """
    Determines the rotation state of an exoplanet from orbital mechanics.

    Usage
    -----
    ra = RotationAnalyser()
    state = ra.full_analysis(a_AU=0.03, M_star=0.08, R_planet=1.0,
                             eccentricity=0.01, system_age_gyr=7.6)
    ra.print_report(state)
    """

    # ------------------------------------------------------------------
    @staticmethod
    def _mass_from_radius(R_earth):
        """Chen & Kipping (2017) mass-radius relation."""
        if R_earth < 1.23:
            return (R_earth / 1.008) ** (1.0 / 0.55) * M_EARTH
        elif R_earth < 14.26:
            return (R_earth / 0.808) ** (1.0 / 0.589) * M_EARTH
        else:
            return (R_earth / 17.74) ** (1.0 / -0.044) * M_EARTH

    # ------------------------------------------------------------------
    def sync_timescale(self, a_AU, M_star_Msun, R_planet_Rearth,
                       eccentricity=0.0, planet_type="rocky",
                       Q=None, k2=None, M_planet_Mearth=None):
        """
        Gladman (1996) synchronization timescale.

        tau_sync = (2*pi*xi/9) * omega0 * M_p * Q * a^6
                   / (G * k2 * M_*^2 * R_p^3)

        where xi = 0.33 (rocky) or 0.25 (gas/ice giant).

        Returns
        -------
        dict  tau_sync_Gyr, tau_sync_yr, parameters
        """
        if Q  is None: Q  = TIDAL_Q.get(planet_type, 100)
        if k2 is None: k2 = LOVE_K2.get(planet_type, 0.299)

        a_m  = a_AU * AU
        M_s  = M_star_Msun * M_SUN
        R_p  = R_planet_Rearth * R_EARTH
        M_p  = (M_planet_Mearth * M_EARTH
                if M_planet_Mearth else self._mass_from_radius(R_planet_Rearth))
        xi   = 0.25 if planet_type in ("gas_giant", "ice_giant") else 0.33

        tau_s = ((2 * np.pi * xi / 9.0)
                 * OMEGA_EARTH * M_p * Q * a_m**6
                 / (G * k2 * M_s**2 * R_p**3))

        return {
            "tau_sync_yr":  tau_s / YEAR_S,
            "tau_sync_Gyr": tau_s / YEAR_S / 1e9,
            "Q": Q, "k2": k2, "xi": xi,
        }

    # ------------------------------------------------------------------
    def pseudo_sync_rate(self, n_orbital_rad_s, eccentricity):
        """
        Hut (1981) pseudo-synchronous rotation rate for eccentric orbits.

        omega_ps / n =
            (1 + (15/2)e^2 + (45/8)e^4 + (5/16)e^6)
            / ((1 + 3e^2 + (3/8)e^4) * (1-e^2)^(3/2))

        For e=0 this reduces to omega_ps = n (synchronous).

        Parameters
        ----------
        n_orbital_rad_s : float   Mean orbital angular velocity (rad/s)
        eccentricity    : float   Orbital eccentricity

        Returns
        -------
        dict  omega_ps, ratio_to_sync, P_rotation_days
        """
        e = eccentricity
        numerator   = (1 + (15/2)*e**2 + (45/8)*e**4 + (5/16)*e**6)
        denominator = (1 + 3*e**2 + (3/8)*e**4) * (1 - e**2)**1.5
        ratio       = numerator / denominator
        omega_ps    = n_orbital_rad_s * ratio
        P_rot_s     = 2 * np.pi / omega_ps if omega_ps > 0 else np.inf
        return {
            "omega_ps_rad_s":     float(omega_ps),
            "ratio_to_sync":      float(ratio),
            "P_rotation_days":    float(P_rot_s / DAY_S),
            "eccentricity":       eccentricity,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def check_resonances(P_orbital_days, P_rotation_days, tolerance=0.05):
        """
        Check for common spin-orbit resonances.

        Tests p:q resonances where omega_rot / n = p:q.
        (e.g. Mercury 3:2, Moon 1:1, Venus retrograde ~-1:243)

        Parameters
        ----------
        P_orbital_days  : float
        P_rotation_days : float
        tolerance       : float  Fractional tolerance for resonance match

        Returns
        -------
        list of dicts — each matching resonance
        """
        resonances = [
            (1, 1, "1:1 synchronous (tidally locked)"),
            (3, 2, "3:2 (Mercury-like)"),
            (2, 1, "2:1"),
            (5, 2, "5:2"),
            (3, 1, "3:1"),
            (1, 2, "1:2 (sub-synchronous)"),
        ]
        ratio = P_orbital_days / P_rotation_days
        matches = []
        for p, q, name in resonances:
            expected = p / q
            if abs(ratio - expected) / expected < tolerance:
                matches.append({
                    "resonance":     name,
                    "p":             p,
                    "q":             q,
                    "expected_ratio": expected,
                    "actual_ratio":  float(ratio),
                    "deviation_pct": float(100 * abs(ratio - expected) / expected),
                })
        return matches

    # ------------------------------------------------------------------
    def sensitivity_grid(self, a_AU, M_star_Msun, R_planet_Rearth,
                          system_age_gyr=5.0):
        """
        Tidal locking probability over a grid of Q and k2 values.

        Returns
        -------
        dict  Q_values, k2_values, locked_fraction, grid (2D array)
        """
        Q_vals  = [10, 50, 100, 500, 1000, 1e4, 1e5]
        k2_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        grid    = np.zeros((len(Q_vals), len(k2_vals)))

        for i, Q in enumerate(Q_vals):
            for j, k2 in enumerate(k2_vals):
                tau = self.sync_timescale(a_AU, M_star_Msun, R_planet_Rearth,
                                          Q=Q, k2=k2)
                ratio = system_age_gyr / tau["tau_sync_Gyr"]
                grid[i, j] = ratio

        locked_fraction = float(np.sum(grid > 1.0) / grid.size)
        return {
            "Q_values":       Q_vals,
            "k2_values":      k2_vals,
            "age_to_tau_grid": grid,
            "locked_fraction": locked_fraction,
        }

    # ------------------------------------------------------------------
    def full_analysis(self, a_AU, M_star_Msun, R_planet_Rearth,
                      eccentricity=0.0, system_age_gyr=5.0,
                      planet_type="rocky", P_orbital_days=None,
                      M_planet_Mearth=None):
        """
        Run the complete rotation state analysis for one planet.

        Returns
        -------
        dict with keys: sync, pseudo_sync, resonances, sensitivity, state_label
        """
        if P_orbital_days is None:
            a_m = a_AU * AU
            M_s = M_star_Msun * M_SUN
            P_orbital_days = (2 * np.pi * np.sqrt(a_m**3 / (G * M_s))) / DAY_S

        tau    = self.sync_timescale(a_AU, M_star_Msun, R_planet_Rearth,
                                     eccentricity=eccentricity,
                                     planet_type=planet_type,
                                     M_planet_Mearth=M_planet_Mearth)
        ratio  = system_age_gyr / tau["tau_sync_Gyr"]
        locked = ratio > 1.0

        n = (2 * np.pi) / (P_orbital_days * DAY_S)
        ps = self.pseudo_sync_rate(n, eccentricity)

        resonances = self.check_resonances(P_orbital_days, ps["P_rotation_days"])

        sensitivity = self.sensitivity_grid(a_AU, M_star_Msun, R_planet_Rearth,
                                            system_age_gyr)

        if locked:
            if abs(eccentricity) < 0.01:
                label = "1:1 Synchronous (Tidally Locked)"
            else:
                label = "Pseudo-synchronous (eccentric lock)"
        elif ratio > 0.1:
            label = "Possibly approaching lock (uncertain)"
        else:
            label = "Freely rotating (not locked)"

        return {
            "sync":            tau,
            "pseudo_sync":     ps,
            "resonances":      resonances,
            "sensitivity":     sensitivity,
            "is_locked":       locked,
            "age_to_tau":      float(ratio),
            "state_label":     label,
            "P_orbital_days":  P_orbital_days,
            "eccentricity":    eccentricity,
            "system_age_gyr":  system_age_gyr,
        }

    def print_report(self, state):
        print("\n" + "="*62)
        print("  ROTATION STATE ANALYSIS")
        print("="*62)
        print(f"  Orbital period:      {state['P_orbital_days']:.4f} days")
        print(f"  Eccentricity:        {state['eccentricity']:.4f}")
        print(f"  System age:          {state['system_age_gyr']:.2f} Gyr")
        print(f"  tau_sync:            {state['sync']['tau_sync_Gyr']:.4g} Gyr")
        print(f"  Age / tau_sync:      {state['age_to_tau']:.4g}")
        print(f"  Pseudo-sync period:  {state['pseudo_sync']['P_rotation_days']:.4f} days")
        print(f"  Pseudo-sync / sync:  {state['pseudo_sync']['ratio_to_sync']:.4f}")
        print(f"  State: {state['state_label']}")
        if state["resonances"]:
            print(f"  Resonance Matches:")
            for r in state["resonances"]:
                print(f"    {r['resonance']} (deviation {r['deviation_pct']:.2f}%)")
        print(f"  Q-k2 sensitivity:    {state['sensitivity']['locked_fraction']*100:.0f}% "
              f"of Q-k2 scenarios predict locking")
        print("="*62)
