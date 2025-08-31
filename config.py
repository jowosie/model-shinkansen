"""
Shinkansen Simulation Parameters
All units in SI unless otherwise stated
"""

# Simulation Controls
SIM_CONTROLS = {
    "dt": 5,
    "sim_time_min": 120,
    "sim_time_sec": 120 * 60,
}

# Route and Infrastructure Parameters
ROUTE_PARAMETERS = {
    "phase1_length": 285600, # length of track for chuo phase 1 implementation
    "phase2_length": 438000, # full length of track for chuo shinkansen
    "testtrack_length": 42800, # length of test track
    "max_gradient": 4,
    "min_curve_radius": 8000,
    "tunnel_fraction": 0.86, # estimated fraction of tunnels for phase 2 line
}

# Vehicle Parameters
LO_VEHICLE = {
    # Mass Properties
    "mass_intermediate_car": 25000,
    "mass_end_car": 28000,
    "num_intermediate_cars": 14,
    "num_end_cars": 2,
    "total_mass_empty": (14 * 25000) + (2 * 28000),
    "passenger_capacity": 1000,
    "avg_passenger_mass": 80,
    "total_mass_loaded": ((14 * 25000) + (2 * 28000)) + (1000 * 80),

    # Geometric Properties
    "width": 2.9,
    "height": 3.1,
    "length_intermediate_car": 24.3,
    "length_end_car": 28,
    "length": (24.3 * 14) + (2 * 28),
    "i_yaw": 1.6e6, # placeholder
    "i_pitch": 8e5, # placeholder
    "i_roll": 2e5, # placeholder

    # Aerodynamic Properties
    "cd_openair": 0.29,
    "tunnel_drag_multiplier": 1.5,
    "frontal_area": 10, # placeholder
}

# Maglev parameters
SCMAGLEV_SYSTEM = {
    "max_speed": 505 / 3.6,
    "levitation_takeoff_speed": 150 / 3.6,
    "lsm_pole_pitch": 1.35,
    "target_avg_acceleration": 0.7,
    "target_avg_deceleration": 0.7,

    # Coil Parameters
    "coil_width": 0.15,
    "coil_length": 0.4,
    "coil_position_z": 10,

    # Levitation Force Parameters
    "max_F_lev": 7.2e6,  # Max levitation force coefficient (N)
    "decay_const": 50,  # Levitation force decay constant (1/m)
    "lev_velocity": 41.67, # Characteristic speed for levitation (m/s)

    # Guidance Force Parameters
    "k_guidance": 2.5e6,  # Guidance stiffness (N/m)
    "c_guidance": 5.0e4,  # Guidance damping (Ns/m)
}

# Physical Constants
CONSTANTS = {
    "gravity": 9.81,
    "air_density": 1.225,
    "mag_permeability": 1.25663706e-6
}