"""
SHINKANSEN PHYSICS ENGINE [v1.0]
Custom built physics engine to model all physics needed to run the maglev
simulation.
"""

import config
import numpy as np
import magpylib as mpl
import magpylib_force as mpf

# Vehicle State (draw from main eventually)
coil_resistance = 0.01
coil_inductance = 0.001
pole_pitch = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]


"""
FORCE MODELING
Calculates all relevant forces acting on the train body while in operation.
"""

# Gravity Force
def calc_gravity_force(gravity, mass):
    """
    Calculates the force due to gravity

    :param gravity: Gravitational acceleration in m/s**2
    :param mass: Total mass of train

    :return: np.ndarray: Gravity force vector [Fx, Fy, Fz] in Newtons
    """

    f_grav = np.zeros(3, dtype=float)

    f_grav[2] = -config.CONSTANTS["gravity"] * config.LO_VEHICLE["total_mass_loaded"]
    return f_grav

# Induced Force
def calc_induced_force(train, guideway, velocity, train_height, lateral_displacement, coil_resistance=0.01, coil_inductance=0.001):
    """
    Calculates the levitation force on the train through an induced current in the guideway coils. Follows
    Faraday's Law of Induction.

    :param train: Train object
    :param guideway: Guideway object
    :param velocity: Current forward velocity of train in m/s
    :param train_height: Vertical position of the train above the guideway in meters
    :param lateral_displacement: Lateral displacement from the guideway center in meters
    :param coil_resistance: Resistance of guideway coil in Ohms
    :param coil_inductance: Inductance of guideway coil in Henrys

    :return: np.ndarray: Total induced force vector [Fx, Fy, Fz] in Newtons
    Fx = Magnetic drag force
    Fy = Guidance force
    Fz = Levitation force
    """

    if velocity < 1:
        return np.zeros(3, dtype=float)

    f_induced = np.zeros(3,dtype=float)
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
                force_i1, torque_i1 = mpf.getFT(magnet, coil1, anchor=(0,0,0))
                force_i2, torque_i2 = mpf.getFT(magnet, coil2, anchor=(0,0,0))
                f_induced += force_i1 + force_i2

    return f_induced

# Propulsion Force
def calc_propulsion_force(train, guideway, velocity, target_velocity):
    """
    Calculates the propulsion force from the LSM modeling the interaction between the train's magnets and the
    actively powered propulsion coils in the guideway.

    :param train: Train object
    :param guideway: Guideway object
    :param velocity: Current forward velocity of train in m/s
    :param target_velocity: Target forward velocity of train in m/s

    :return: np.ndarray: Total propulsion force vector [Fx, Fy, Fz] in Newtons
    """

    f_propulsion = np.zeros(3, dtype=float)

    error = target_velocity - velocity
    current_gain = 100_000
    max_current = np.clip(error * current_gain, -5000, 5000)

    train_magnets = train.get_magnets()
    propulsion_coils = guideway.get_propulsion_coils()

    lsm_wavelength = 2.0 * pole_pitch

    for magnet in train_magnets:
        for coil in propulsion_coils:
            # Only consider nearby coils
            if abs(magnet.position[0] - coil.position[0]) < 3.0:

                # Create traveling magnetic wave using phase difference between magnet
                # position and coil position.
                phase = (2 * np.pi / lsm_wavelength) * (coil.position[0] - magnet.position[0])

                magnet_polarity = np.sign(magnet.orientation.as_rotvec()[2])

                coil.current = magnet_polarity * max_current * np.sin(phase + np.pi/2)

                force_p, torque_p = mpf.getFT(magnet, coil, anchor=(0,0,0))
                f_propulsion += force_p

    return f_propulsion


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

    f_aero_drag = np.zeros(3, dtype=float)

    density = config.CONSTANTS["air_density"]
    drag_coeff = config.LO_VEHICLE["cd_openair"]
    frontal_area = config.LO_VEHICLE["frontal_area"]

    f_aero_drag[0] = -0.5 * density * velocity**2 * drag_coeff * frontal_area
    return f_aero_drag