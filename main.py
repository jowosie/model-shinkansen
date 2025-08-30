import matplotlib as mpl
import matplotlib.pyplot as plt
from physics import Train
from controller import PIDController

"""
Main Loop
"""

# Simulation Setup
sim_time =      120         # seconds
dt =            0.01        # time step
total_steps =   int(sim_time / dt)

# Create Objects
shinkansen = Train(mass_kg = 700000, drag_coeff = 0.25)

# Create PID Controller
pid = PIDController(Kp=150000, Ki=1, Kd=1, setpoint=80)

# Data Logging
time_log =      []
velocity_log =  []
position_log =  []
force_log =     []
setpoint_log =  []

# Simulation Loop
for step in range(total_steps):
    # Calculate total time
    current_time = step * dt

    # Obtain output from PID Controller
    prop_force = pid.calculate(shinkansen.velocity, dt)

    # Feed force into physics engine
    shinkansen.update_state(prop_force, dt)

    # Log data
    time_log.append(current_time)
    velocity_log.append(shinkansen.velocity)
    position_log.append(shinkansen.position)
    force_log.append(prop_force)
    setpoint_log.append(pid.setpoint)

# Plot Results
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

# Velocity vs. Time
ax1.plot(time_log, velocity_log, label='Train Velocity')
ax1.plot(time_log, setpoint_log, 'r--', label='Target Velocity')
ax1.set_ylabel('Velocity (m/s)')
ax1.legend()
ax1.grid(True)

# Force vs. Time
ax2.plot(time_log, force_log, label='Propulsion Force')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force (N)')
ax2.legend()
ax2.grid(True)

plt.suptitle('Maglev Shinkansen Simulation')
plt.show()