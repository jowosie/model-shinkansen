import numpy as np

class Train:
    def __init__(self, mass_kg, drag_coeff):
        self.mass_kg =      mass_kg
        self.drag_coeff =   drag_coeff

        # Initial State
        self.position =     0.0     # m
        self.velocity =     0.0     # m/s
        self.acceleration = 0.0     # m/s^2

    def update_state(self, prop_force, dt):
        # Calculate Drag Force
        drag_force = self.drag_coeff * self.velocity**2 * np.sign(-self.velocity)

        # Calculate Net Force
        net_force = prop_force + drag_force

        # Calculate Acceleration
        self.acceleration = net_force / self.mass_kg

        # Update Velocity and Position
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt