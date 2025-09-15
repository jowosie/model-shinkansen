print("\n\n >>>>> LOADING PHYSICS.PY <<<<<\n\n")

"""
SHINKANSEN PHYSICS ENGINE [v1.1]
Custom built physics engine to model all physics needed to run the maglev
simulation.
"""

import config
import numpy as np
import magpylib as mpl
import magpylib_force as mpf
from train import Train
from guideway import Guideway

# Vehicle State (draw from main eventually)
coil_resistance = 0.01
coil_inductance = 0.001
pole_pitch = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]


"""
FORCE MODELING
Calculates all relevant forces acting on the train body while in operation.
"""


# Gravity Force
def calc_gravity_force(mass: float) -> np.ndarray:
    """
    Calculates the force due to gravity.

    :param mass: Total mass of train
    :return: np.ndarray: Gravity force vector [Fx, Fy, Fz] in Newtons
    """
    f_grav = np.zeros(3, dtype=float)
    f_grav[2] = -mass * config.CONSTANTS["gravity"]
    return f_grav


# Induced Force
def calc_induced_force(
    train: Train,
    guideway: Guideway,
    velocity: float,
    train_height: float,
    lateral_displacement: float,
    coil_resistance: float = 0.01,
    coil_inductance: float = 0.001,
) -> np.ndarray:
    """
    Calculates the levitation force on the train through an induced current in the guideway coils. Follows
    Faraday's Law of Induction.
    """
    if abs(velocity) < 1.0:
        return np.zeros(3, dtype=float)

    f_induced = np.zeros(3, dtype=float)
    train_magnets = train.get_magnets()
    guideway_coils = guideway.get_levitation_coils()

    for magnet in train_magnets.sources:
        for coil_pair_index in range(0, len(guideway_coils.sources), 2):
            coil1 = guideway_coils.sources[coil_pair_index]
            coil2 = guideway_coils.sources[coil_pair_index + 1]

            if abs(magnet.position[0] - coil1.position[0]) < 2:
                B = mpl.getB(magnet, coil1.position)  # Tesla

                delta_x = 0.01  # small step (1cm)
                B_dx = mpl.getB(magnet, np.add(coil1.position, (delta_x, 0, 0)))

                dB_dx = (B_dx - B) / delta_x
                dB_dt = velocity * dB_dx

                coil_area = np.pi * (coil1.diameter / 2) ** 2
                emf = -dB_dt[2] * coil_area

                omega = 2 * np.pi * abs(velocity) / pole_pitch
                impedance = np.sqrt(
                    coil_resistance**2 + (omega * coil_inductance) ** 2
                )

                induced_current = emf / impedance if impedance > 0 else 0

                coil1.current = induced_current
                coil2.current = -induced_current

                force_i1, _ = mpf.getFT(magnet, coil1, anchor=(0, 0, 0))
                force_i2, _ = mpf.getFT(magnet, coil2, anchor=(0, 0, 0))
                f_induced += force_i1 + force_i2

    return f_induced


# Propulsion Force
def calc_propulsion_force(
    train: Train, guideway: Guideway, velocity: float, target_velocity: float
) -> np.ndarray:
    """
    Calculates the propulsion force from the LSM.
    """
    f_propulsion = np.zeros(3, dtype=float)

    # Cleaned up: Get collections once at the start
    train_magnets = train.get_magnets()
    propulsion_coils = guideway.get_propulsion_coils()

    # Reset all coil currents to 0
    for coil in propulsion_coils.sources:
        coil.current = 0

    error = target_velocity - velocity
    current_gain = config.SCMAGLEV_SYSTEM["current_gain"]
    max_current = np.clip(error * current_gain, -5000, 5000)

    lsm_wavelength = 2.0 * pole_pitch

    for magnet in train_magnets.sources:
        for coil in propulsion_coils.sources:
            # Calculate forces only for nearby coils
            if abs(magnet.position[0] - coil.position[0]) < 3.0:
                phase = (2 * np.pi / lsm_wavelength) * (
                    coil.position[0] - magnet.position[0]
                )

                magnet_polarity = np.sign(magnet.magnetization[2])

                coil.current = magnet_polarity * max_current * np.sin(phase + np.pi / 2)

                force_p, _ = mpf.getFT(magnet, coil, anchor=(0, 0, 0))

                f_propulsion += force_p

    return f_propulsion


# Aerodynamic Drag
def calc_aero_drag_force(
    velocity: float, density: float, drag_coeff: float, frontal_area: float
) -> np.ndarray:
    """
    Calculates the aerodynamic drag force on the train.
    """
    f_aero_drag = np.zeros(3, dtype=float)
    # Drag force opposes the direction of velocity
    f_aero_drag[0] = -0.5 * density * velocity * abs(velocity) * drag_coeff * frontal_area
    return f_aero_drag