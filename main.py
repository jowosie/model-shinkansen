"""
SHINKANSEN SIMULATOR MAIN SIMULATION LOOP
Main simulation loop that takes the physics and runs a simulation over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from train import Train
from guideway import Guideway
from physics import (
    calc_gravity_force,
    calc_induced_force,
    calc_propulsion_force,
    calc_aero_drag_force,
)
import config


def derivatives(t, y, train, guideway, target_velocity):
    """
    Calculates the derivatives of the state vector dy/dt = [vx, vy, vz, ax, ay, az].

    :param t: Current time
    :param y: State vector [x, y, z, vx, vy, vz]
    :param train: Train object
    :param guideway: Guideway object
    :param target_velocity: Target velocity for the train
    :return: An array of derivatives [vx, vy, vz, ax, ay, az]
    """
    position = y[:3]
    velocity = y[3:]

    # Update the train's physical position in the magpylib model
    train.set_position(position)

    # Calculate Forces
    mass = config.LO_VEHICLE["total_mass_loaded"]
    air_density = config.CONSTANTS["air_density"]

    induced_forces = calc_induced_force(train, guideway, velocity[0], position[2], position[1])
    print(induced_forces)
    propulsion_forces = calc_propulsion_force(train, guideway, velocity[0], target_velocity)
    print(propulsion_forces)
    gravitational_forces = calc_gravity_force(mass)
    print(gravitational_forces)
    aero_drag_forces = calc_aero_drag_force(velocity[0],air_density,config.LO_VEHICLE["cd_openair"],config.LO_VEHICLE["frontal_area"],)
    print(aero_drag_forces)

    total_forces = np.sum([induced_forces, propulsion_forces, gravitational_forces, aero_drag_forces], axis=0)

    print(total_forces)

    # Calculate acceleration
    acceleration = total_forces / mass

    # Return the derivatives
    return np.concatenate((velocity, acceleration))


def rk4_step(t, y, dt, train, guideway, target_velocity):
    """
    Performs a single step of the RK4 method.
    """
    k1 = derivatives(t, y, train, guideway, target_velocity)
    k2 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k1, train, guideway, target_velocity)
    k3 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k2, train, guideway, target_velocity)
    k4 = derivatives(t + dt, y + dt * k3, train, guideway, target_velocity)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulation():
    """
    Main function to run the simulation.
    """

    # Simulation Parameters
    simulation_duration = 60  # s
    time_step = 0.1  # s
    target_velocity = 138  # m/s

    # Initialize Model Objects
    print("Initializing train and guideway models...")
    train = Train(num_bogies=2, magnets_per_bogie=8)
    guideway = Guideway(length=2000)

    # Initialize State Variables
    # State vector y = [x, y, z, vx, vy, vz]
    y = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])

    # History for plotting
    time_history = []
    velocity_history = []
    position_history = []
    force_history = []

    # Simulation Loop
    print("Starting simulation...")
    for t in np.arange(0, simulation_duration, time_step):
        try:
            y = rk4_step(t, y, time_step, train, guideway, target_velocity)

            # Constraints
            # Prevents the train from falling through the guideway
            if y[2] < 0.02 and y[5] < 0:
                y[2] = 0.02  # z position
                y[5] = 0  # vz velocity

            # Record Data
            time_history.append(t)
            velocity_history.append(y[3])  # Stores forward velocity (vx)
            position_history.append(y[2])  # Stores vertical position (z)
            force_history.append(
                derivatives(t, y, train, guideway, target_velocity)[5]
                * config.LO_VEHICLE["total_mass_loaded"]
            )  # Stores levitation forces (F_z = m*a_z)

            if int(t) % 5 == 0:
                print(
                    f"Time: {t:.1f}s, Speed: {y[3] * 3.6:.1f} km/h, Levitation gap: {y[2] * 100:.2f} cm"
                )
        except Exception as e:
            print(f"An error occurred at t={t:.1f}s: {e}")
            break

    print(f"Simulation finished. Recorded {len(time_history)} timesteps.")
    return time_history, velocity_history, position_history, force_history


def plot_results(time, velocity, position, lev_force):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

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