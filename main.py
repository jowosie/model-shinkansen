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
    calc_wheel_support_force
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

    # Initialize all forces to zero arrays to ensure they're never None
    induced_forces = np.zeros(3, dtype=float)
    propulsion_forces = np.zeros(3, dtype=float)
    gravitational_forces = np.zeros(3, dtype=float)
    aero_drag_forces = np.zeros(3, dtype=float)
    wheel_support_forces = np.zeros(3, dtype=float)

    try:
        # Calculate Forces
        mass = config.LO_VEHICLE["total_mass_loaded"]
        air_density = config.CONSTANTS["air_density"]

        # Calculate each force component - each function should always return a numpy array
        induced_forces = calc_induced_force(train, guideway, velocity[0], position[2], position[1])
        propulsion_forces = calc_propulsion_force(train, guideway, velocity[0], target_velocity, position[0], t)
        gravitational_forces = calc_gravity_force(mass)
        aero_drag_forces = calc_aero_drag_force(
            velocity[0],
            air_density,
            config.LO_VEHICLE["cd_openair"],
            config.LO_VEHICLE["frontal_area"],
        )

        # Validate that all forces are numpy arrays
        for force_name, force in [
            ("induced", induced_forces),
            ("propulsion", propulsion_forces),
            ("gravitational", gravitational_forces),
            ("aero_drag", aero_drag_forces)
        ]:
            if force is None:
                print(f"WARNING: {force_name}_forces returned None, using zero vector")
                if force_name == "induced":
                    induced_forces = np.zeros(3, dtype=float)
                elif force_name == "propulsion":
                    propulsion_forces = np.zeros(3, dtype=float)
                elif force_name == "gravitational":
                    gravitational_forces = np.zeros(3, dtype=float)
                elif force_name == "aero_drag":
                    aero_drag_forces = np.zeros(3, dtype=float)
            elif not isinstance(force, np.ndarray):
                print(f"WARNING: {force_name}_forces returned {type(force)}, converting to numpy array")
                if force_name == "induced":
                    induced_forces = np.array(force, dtype=float).reshape(3)
                elif force_name == "propulsion":
                    propulsion_forces = np.array(force, dtype=float).reshape(3)
                elif force_name == "gravitational":
                    gravitational_forces = np.array(force, dtype=float).reshape(3)
                elif force_name == "aero_drag":
                    aero_drag_forces = np.array(force, dtype=float).reshape(3)

        # Pass only the z-components (scalars) to wheel support calculation
        if induced_forces is not None:
            levitation_z = float(induced_forces[2])
        else:
            levitation_z = 0.0

        if gravitational_forces is not None:
            gravity_z = float(gravitational_forces[2])
        else:
            gravity_z = 0.0

        wheel_support_forces = calc_wheel_support_force(
            velocity[0],
            levitation_z,
            gravity_z
        )

        # Ensure wheel_support_forces is also valid
        if wheel_support_forces is None:
            print("WARNING: wheel_support_forces returned None, using zero vector")
            wheel_support_forces = np.zeros(3, dtype=float)

        # Sum all forces
        total_forces = induced_forces + propulsion_forces + gravitational_forces + aero_drag_forces + wheel_support_forces

        # Calculate acceleration
        acceleration = total_forces / mass

        # Return the derivatives
        return np.concatenate((velocity, acceleration))

    except Exception as e:
        print("\n" + "="*20 + " DEBUG SNAPSHOT " + "="*20)
        print(f"Caught an exception at simulation time t = {t:.4f}s")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\n--- STATE VARIABLES ---")
        print(f"Position (y[:3]): {position}")
        print(f"Velocity (y[3:]): {velocity}")
        print("\n--- FORCE COMPONENTS (at time of error) ---")
        print(f"induced_forces       | value: {induced_forces}\t| type: {type(induced_forces)}")
        print(f"propulsion_forces    | value: {propulsion_forces}\t| type: {type(propulsion_forces)}")
        print(f"gravitational_forces | value: {gravitational_forces}\t| type: {type(gravitational_forces)}")
        print(f"aero_drag_forces     | value: {aero_drag_forces}\t| type: {type(aero_drag_forces)}")
        print(f"wheel_support_forces | value: {wheel_support_forces}\t| type: {type(wheel_support_forces)}")
        print("="*58 + "\n")
        # Re-raise the error to stop the simulation
        raise e


def rk4_step(t, y, dt, train, guideway, target_velocity):
    """
    Performs a single step of the RK4 method.
    """
    print(f"  Calculating RK4 step 1/4 for time t={t:.2f}s...")
    k1 = derivatives(t, y, train, guideway, target_velocity)

    print(f"  Calculating RK4 step 2/4 for time t={t:.2f}s...")
    k2 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k1, train, guideway, target_velocity)

    print(f"  Calculating RK4 step 3/4 for time t={t:.2f}s...")
    k3 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k2, train, guideway, target_velocity)

    print(f"  Calculating RK4 step 4/4 for time t={t:.2f}s...")
    k4 = derivatives(t + dt, y + dt * k3, train, guideway, target_velocity)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulation():
    """
    Main function to run the simulation.
    """

    # Simulation Parameters
    simulation_duration = config.SIM_CONTROLS["sim_time_min"] * 60  # s
    time_step = config.SIM_CONTROLS["dt"]  # s
    target_velocity = 138  # m/s

    # Initialize Model Objects
    print("\n >>>>> INITIALIZING TRAIN AND GUIDEWAY MODELS <<<<<\n")
    train = Train(num_bogies=2, magnets_per_bogie=8)
    guideway = Guideway(length=50)

    # Initialize State Variables
    # State vector y = [x, y, z, vx, vy, vz]
    y = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])

    # History for plotting
    time_history = []
    velocity_history = []
    position_history = []
    force_history = []

    on_wheels = True # Flag to tell whether train is on wheels or not

    # Simulation Loop
    print("\n >>>>> STARTING SIMULATION <<<<<\n")
    for t in np.arange(0, simulation_duration, time_step):
        try:
            y = rk4_step(t, y, time_step, train, guideway, target_velocity)

            if on_wheels and y[3] >= config.SCMAGLEV_SYSTEM["levitation_takeoff_speed"]:
                on_wheels = False
                print(" >>> TAKEOFF SPEED REACHED -- LEVITATION SYSTEM ENGAGED <<<")

            # Prevents the train from falling through the guideway when on wheels
            if on_wheels and y[2] < 0.1:
                y[2] = 0.1  # z position
                if y[5] < 0:
                    y[5] = 0  # vz velocity

            # Record Data
            time_history.append(t)
            velocity_history.append(y[3])  # Stores forward velocity (vx)
            position_history.append(y[2])  # Stores vertical position (z)

            # Calculate force for history (with error checking)
            deriv_result = derivatives(t, y, train, guideway, target_velocity)
            if deriv_result is not None and len(deriv_result) >= 6:
                force_history.append(deriv_result[5] * config.LO_VEHICLE["total_mass_loaded"])
            else:
                force_history.append(0.0)  # Default to zero if calculation fails

            if t % 1 == 0:
                print(
                    f"Time: {t:.1f}s, Speed: {y[3] * 3.6:.1f} km/h, Levitation gap: {y[2] * 100:.2f} cm"
                )
        except Exception as e:
            print(f"An error occurred at t={t:.1f}s: {e}")
            print(f"Current state: position={y[:3]}, velocity={y[3:]}")
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