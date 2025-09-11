"""
SHINKANSEN SIMULATOR MAIN SIMULATION LOOP
Main simulation loop that takes the physics and runs a simulation over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from train import Train
from guideway import Guideway
from physics import calc_gravity_force, calc_induced_force, calc_propulsion_force, calc_aero_drag_force
import config

def simulation():
    """
    Main function to run the simulation
    """

    # Simulation Parameters
    simulation_duration = 60 # s
    time_step = 0.1 # s
    target_velocity = 138 # m/s

    # Initialize Model Objects
    print("Initializing train and guideway models...")
    train = Train(num_bogies=2, magnets_per_bogie=8)
    guideway = Guideway(length=2000)

    # Initialize State Variables
    position = np.array([0.0, 0.0, 0.02])
    velocity = np.array([0.0, 0.0, 0.0])
    acceleration = np.array([0.0, 0.0, 0.0])

    # History for plotting
    time_history = []
    velocity_history = []
    position_history = []
    force_history = []

    # Constants
    mass = config.LO_VEHICLE["total_mass_loaded"]
    gravity = config.CONSTANTS["gravity"]
    air_density = config.CONSTANTS["air_density"]

    # Simulation Loop
    print("Starting simulation...")
    for t in np.arange(0, simulation_duration, time_step):
        try:
            # Update train's physical position in magpylib model
            train.set_position(position)

            # Calculate Forces
            induced_forces = calc_induced_force(train, guideway, velocity[0], position[2], position[1])
            propulsion_forces = calc_propulsion_force(train, guideway, velocity[0], target_velocity)
            gravitational_forces = calc_gravity_force(gravity, mass)
            aero_drag_forces = calc_aero_drag_force(velocity[0], air_density, config.LO_VEHICLE["cd_openair"], config.LO_VEHICLE["frontal_area"])

            total_forces = induced_forces + propulsion_forces + gravitational_forces + aero_drag_forces

            total_forces = np.array(total_forces)

            acceleration = total_forces / mass

            # Numerical integration
            velocity += acceleration * time_step
            position += velocity * time_step

            # Constraints
            # Prevents the train from falling through the guideway
            if position[2] < 0.02 and velocity[2] < 0:
                position[2] = 0.02
                velocity[2] = 0

                if acceleration[2] < 0:
                    acceleration[2] = 0

            # Record Data
            time_history.append(t)
            velocity_history.append(velocity[0]) # Stores forward velocity
            position_history.append(position[2]) # Stores vertical position
            force_history.append(total_forces[2]) # Stores levitation forces

            if int(t) % 5 == 0:
                print(f"Time: {t:.1f}s, Speed: {velocity[0] * 3.6:.1f} km/h, Levitation gap: {position[2] * 100:.2f} cm")
        except Exception as e:
            print(f"An error occurred at t={t:.1f}s: {e}")
            break

    print(f"Simulation finished. Recorded {len(time_history)} timesteps.")
    return time_history, velocity_history, position_history, force_history

def plot_results(time, velocity, position, lev_force):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 15))

    # Plot 1 - Velocity vs. Time
    ax1.plot(time, np.array(velocity) * 3.6)
    ax1.set_title("Train velocity vs. Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (km/h)")
    ax1.grid(True)

    # Plot 2 - Levitation Gap vs. Time
    ax2.plot(time, np.array(position) * 100)
    ax2.set_title("Levitation Gap vs. Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Levitation Gap (cm)")
    ax2.grid(True)

    # Plot 3 - Levitation Force vs. Time
    ax3.plot(time, lev_force)
    ax3.set_title("Levitation Force vs. Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Levitation Force (N)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the simulation
    time_hist, vel_hist, pos_hist, force_hist = simulation()

    # Plot the results
    if time_hist:
        plot_results(time_hist, vel_hist, pos_hist, force_hist)
