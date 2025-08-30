"""
SHINKANSEN PHYSICS ENGINE [v0.1]
Custom built physics engine to model all physics needed to run the maglev
simulation.
"""

import config
import numpy as np
import magpylib as mpl

# Vehicle State (draw from main eventually)
current_position = 0
current_velocity = 0
current_acceleration = 0

# FORCE MODELING
F_total = np.empty([3,4], dtype=np.float64) # empty matrix for total force calculations

# Gravity Force
F_grav = -config.CONSTANTS["gravity"] * config.LO_VEHICLE["total_mass_loaded"]
print(F_grav)

# Vehicle magnets currently have placeholder values. Update with real world information
# as it is obtained through research :)

# Superconducting Magnets
sc_magnet = mpl.magnet.Cuboid(
    polarization=(0,0,1.5),
    dimension=(0.5,0.2,0.1)
)

vehicle_magnets = mpl.Collection(sc_magnet)

# Guideway Coil
coil_vertices = np.array([0,3])
coil_segments = np.array([[0, 1], [1, 2], [2, 3]])

# Levitation Force
while current_velocity <= config.SCMAGLEV_SYSTEM["levitation_takeoff_speed"]:
    F_lev = 0
    break # temp
# Guidance Force
# Magnetic Drag Force
# Maglev Force

# Thrust Force
# Normal Force
# Propulsion Force

# Aerodynamic Drag
F_aero_drag = 0.5 * current_velocity**2 * config.CONSTANTS["air_density"] * config.LO_VEHICLE["cd_openair"] * config.LO_VEHICLE["frontal_area"]
print(F_aero_drag)