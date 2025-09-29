print("\n >>>>> LOADING PHYSICS.PY <<<<<\n")

"""
SHINKANSEN PHYSICS ENGINE [v1.4]
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
    # Return zero force if velocity is negligible to avoid instability at low speeds
    if abs(velocity) < 1.0:
        return np.zeros(3, dtype=float)

    f_induced = np.zeros(3, dtype=float)
    train_magnets = train.get_magnets()
    guideway_coils = guideway.get_levitation_coils()

    for magnet in train_magnets.sources:
        for coil_pair_index in range(0, len(guideway_coils.sources), 2):
            coil1 = guideway_coils.sources[coil_pair_index]
            coil2 = guideway_coils.sources[coil_pair_index + 1]

            # Only calculate for nearby coils to improve performance
            if abs(magnet.position[0] - coil1.position[0]) < 2:
                # --- Start of Robustness Changes ---

                # 1. Get B-field values and check if they are valid
                B = mpl.getB(magnet, coil1.position)
                if B is None:
                    continue  # Skip this coil interaction if B-field is None

                delta_x = 0.01
                B_dx = mpl.getB(magnet, np.add(coil1.position, (delta_x, 0, 0)))
                if B_dx is None:
                    continue  # Skip this coil interaction if B-field is None

                # 2. Proceed with calculations now that inputs are validated
                dB_dx = (B_dx - B) / delta_x
                dB_dt = velocity * dB_dx

                coil_area = np.pi * (coil1.diameter / 2) ** 2
                emf = -dB_dt[2] * coil_area

                omega = 2 * np.pi * abs(velocity) / pole_pitch
                impedance = np.sqrt(
                    coil_resistance ** 2 + (omega * coil_inductance) ** 2
                )

                induced_current = emf / impedance if impedance > 0 else 0

                coil1.current = induced_current
                coil2.current = -induced_current

                # 3. Safely get Force-Torque values and unpack only if valid
                ft1 = mpf.getFT(magnet, coil1, anchor=(0, 0, 0))
                ft2 = mpf.getFT(magnet, coil2, anchor=(0, 0, 0))

                if ft1 is not None and ft2 is not None:
                    force_i1, _ = ft1
                    force_i2, _ = ft2

                    # Final check to ensure the force components are valid arrays
                    if isinstance(force_i1, np.ndarray) and isinstance(force_i2, np.ndarray):
                        f_induced += force_i1 + force_i2

    return f_induced


def calc_propulsion_force(
    train: Train, guideway: Guideway, velocity: float, target_velocity: float,
    train_x_position: float, time: float = 0
) -> np.ndarray:
    """
    Calculates the propulsion force from the LSM.
    """
    f_propulsion = np.zeros(3, dtype=float)
    train_magnets = train.get_magnets()
    propulsion_coils = guideway.get_propulsion_coils()

    # Determine the control current based on the speed error
    error = target_velocity - velocity
    current_gain = config.SCMAGLEV_SYSTEM["current_gain"]

    # Use a base current even at zero velocity to start the train
    base_current = 1000  # Base current to ensure the train can start
    max_current = np.clip(error * current_gain + base_current, -5000, 5000)

    lsm_wavelength = 2.0 * pole_pitch

    # Calculate slip frequency for asynchronous operation
    # This ensures the field moves faster than the train to produce thrust
    slip_velocity = 5.0  # m/s - field should move 5 m/s faster than train
    field_velocity = velocity + slip_velocity if error > 0 else velocity - slip_velocity

    # Time-based phase for creating traveling wave
    time_phase = time  # Use actual simulation time

    # 1. Set the currents for all coils to create a traveling magnetic wave
    for coil in propulsion_coils.sources:
        # Calculate phase based on position and desired traveling wave
        spatial_phase = (2 * np.pi / lsm_wavelength) * coil.position[0]

        # Phase shift to create traveling wave effect
        # The wave should "lead" the train position to pull it forward
        travel_phase = (2 * np.pi / lsm_wavelength) * (field_velocity * time_phase - train_x_position)

        total_phase = spatial_phase - travel_phase

        # Apply three-phase currents (assuming three-phase system)
        # Determine which phase this coil belongs to based on position
        phase_group = int(coil.position[0] / (lsm_wavelength / 3)) % 3
        phase_offset = phase_group * (2 * np.pi / 3)

        coil.current = max_current * np.sin(total_phase + phase_offset)

    # 2. Calculate the total force on the train
    for magnet in train_magnets.sources:
        for coil in propulsion_coils.sources:
            # Only calculate for nearby coils to save computation
            distance = abs(magnet.position[0] - coil.position[0])
            if distance < 3.0:
                # Calculate the force ON the magnet FROM the coil
                # Note: The force direction might need to be reversed depending on
                # the magpylib convention
                force_p, _ = mpf.getFT(magnet, coil, anchor=(0, 0, 0))

                # Add a distance-based scaling factor for more realistic force distribution
                force_scale = np.exp(-distance / 2.0)
                f_propulsion += force_p * force_scale

    # Add a startup assistance force if velocity is very low
    # This helps overcome initial static friction and gets the train moving
    if abs(velocity) < 1.0 and error > 0:
        startup_force = 50000 * error  # Proportional startup assistance
        f_propulsion[0] += startup_force

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


def calc_wheel_support_force(
        velocity: float, levitation_force: float, gravitational_force: float
) -> np.ndarray:
    """
    Calculates the support force from the wheels at low speeds.
    """

    f_wheel_support = np.zeros(3, dtype=float)
    takeoff_speed = config.SCMAGLEV_SYSTEM["levitation_takeoff_speed"]  # m/s

    if abs(velocity) < takeoff_speed:
        # Only apply support force if the net force is downwards
        net_vertical_force = float(levitation_force) + float(gravitational_force)

        if net_vertical_force < 0:
            f_wheel_support[2] = -net_vertical_force

    return f_wheel_support