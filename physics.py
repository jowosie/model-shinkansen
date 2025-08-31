"""
SHINKANSEN PHYSICS ENGINE [v1.0]
Custom built physics engine to model all physics needed to run the maglev
simulation.
"""

import config
import numpy as np
import magpylib as mpl
import train
import guideway

# Vehicle State (draw from main eventually)
current_position = 0
current_velocity = 0
current_acceleration = 0
levitation_gap = 0.1 # Placeholder for dynamic levitation gap (m)
lateral_displacement = 0.01 # Placeholder for lateral displacement from guideway center (m)
lateral_velocity = 0 # Placeholder for lateral velocity (m/s)

"""
FORCE MODELING
Calculates all relevant forces acting on the train body while in operation.
"""

F_total = np.empty([3,4], dtype=np.float64) # empty matrix for total force calculations

# Gravity Force
def calc_gravity_force(gravity, mass):
    """
    Calculates the force due to gravity

    :param gravity: Gravitational acceleration in m/s**2
    :param mass: Total mass of train

    :return: np.ndarray: Gravity force vector [Fx, Fy, Fz] in Newtons
    """

    f_grav = np.zeros(3)

    f_grav[1] = -config.CONSTANTS["gravity"] * config.LO_VEHICLE["total_mass_loaded"]
    return f_grav

# Levitation Force
def calc_levitation_force(train, guideway, velocity, train_height, coil_resistance=0.01, coil_inductance=0.001):
    """
    Calculates the levitation force on the train through an induced current in the guideway coils. Follows
    Faraday's Law of Induction.

    :param train: Train object
    :param guideway: Guideway object
    :param velocity: Current forward velocity of train in m/s
    :param train_height: Vertical position of the train above the guideway in meters
    :param coil_resistance: Resistance of guideway coil in Ohms
    :param coil_inductance: Inductance of guideway coil in Henrys

    :return: np.ndarray: Total levitation force vector [Fx, Fy, Fz] in Newtons
    """

    if velocity < 1:
        return np.zeros(3)

    f_levitation = np.zeros(3)
    train_magnets = train.get_magnets()
    guideway_coils = guideway.get_levitation_coils()

    for magnet in train_magnets:
        for coil_pair_index in range(0, len(guideway_coils.sources), 2):
            coil1 = guideway_coils.sources[coil_pair_index]
            coil2 = guideway_coils.sources[coil_pair_index + 1]

            if abs(magnet.position[0] - coil1.position[0]) < 2:
                B = mpl.getB(magnet, coil1.position) # Tesla

                delta_x = 0.01 # small step (1cm)
                B_dx = mpl.getB(magnet, np.add(coil1.position, (delta_x, 0, 0)))

                # Numerical approximation of flux derivative with respect to time
                dB_dx = (B_dx - B) / delta_x
                dB_dt = velocity * dB_dx

                coil_area = np.pi * (coil1.diameter / 2)**2
                emf = -dB_dt[2] * coil_area # Uses z-component of B-field change

                pole_pitch = 1.0
                omega = 2 * np.pi * velocity / pole_pitch
                impedance = np.sqrt(coil_resistance**2 + (omega * coil_inductance)**2)

                if impedance > 0:
                    induced_current = emf / impedance
                else:
                    induced_current = 0

                # Set current on coils based on induced current
                coil1.current = induced_current
                coil2.current = -induced_current

                # Calculate Lorentz Force on the figure-eight coil using magpylib
                lorentz_force = coil1.getF(magnet) + coil2.getF(magnet)
                f_levitation += lorentz_force

    return f_levitation

# Guidance Force


# Magnetic Drag Force
# Maglev Force

# Thrust Force
# Normal Force
# Propulsion Force

# Aerodynamic Drag
def calc_aero_drag_force(velocity, density, drag_coeff, frontal_area):
    """
    Calculates the aerodynamic drag force on the train at the current velocity.

    :param velocity: Current forward velocity of the train in m/s
    :param density: Air density in kg/m^3
    :param drag_coeff: Drag coefficient of the train
    :param frontal_area: Frontal area of the train in m**2

    :return: np.ndarray: Aerodynamic drag force vector [Fx, Fy, Fz] in Newtons
    """

    f_aero_drag = np.zeros(3)

    density = config.CONSTANTS["air_density"]
    drag_coeff = config.LO_VEHICLE["cd_openair"]
    frontal_area = config.LO_VEHICLE["frontal_area"]

    f_aero_drag[0] = -0.5 * density * velocity**2 * drag_coeff * frontal_area