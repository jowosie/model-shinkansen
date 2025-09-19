"""
SHINKANSEN TRAIN DEFINITION
Defines the train bodies and SCMs on the train
"""

import magpylib as mpl
import config

class Train:
    def __init__(self, num_bogies=2, magnets_per_bogie=4, magnet_strength=1_000_000):
        """
        Initializes the train with bogies and SCMs

        :param num_bogies (int): Number of bogies on the train
        :param magnets_per_bogie (int): Number of magnets per bogie
        :param magnet_strength (float): Magnetic dipole moment of SCMs (A*m^2)
        """
        self.num_bogies = num_bogies
        self.magnets_per_bogie = magnets_per_bogie
        self.magnet_strength = magnet_strength
        self.bogies = self.create_bogies()

    # In train.py

    def create_bogies(self):
        """Creates the magnet arrangement for the train's bogies."""
        all_magnets = []
        # Distance between the center of one bogie and the next
        bogie_separation = 20  # meters

        # The y-distance from the centerline to the magnets on each side
        magnet_y_spacing = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]

        # Number of magnet pairs (N-S) per side of a bogie
        num_magnet_pairs_per_side = self.magnets_per_bogie // 2

        for i in range(self.num_bogies):
            # Loop for each side of the bogie (-1 for left, 1 for right)
            for side in [-1, 1]:
                # Loop through each magnet position along the length of the bogie
                for j in range(self.magnets_per_bogie):
                    # Polarity alternates along the x-axis (j)
                    polarity = 1 if j % 2 == 0 else -1

                    # Magnets are spaced by the pole pitch along the x-axis
                    x_pos = (i * bogie_separation) + (j * config.SCMAGLEV_SYSTEM["lsm_pole_pitch"])

                    # Y-position is constant for each side
                    y_pos = side * magnet_y_spacing / 2

                    # Z-position is constant
                    z_pos = 0

                    magnet = mpl.magnet.Cuboid(
                        magnetization=(0, 0, polarity * self.magnet_strength),
                        dimension=(0.5, 0.2, 0.4),  # meters (length, width, height)
                        position=(x_pos, y_pos, z_pos)
                    )
                    all_magnets.append(magnet)

                    magnet.meshing = (5, 5, 5)

        # Creates a single collection for all the magnets on the whole train
        return mpl.Collection(all_magnets)

    def get_magnets(self):
        """Returns the magpylib collection of all magnets on the train."""
        return self.bogies

    def set_position(self, position):
        """
        Set the position of the entire train.

        :param position (tuple or list): (x, y, z) coordinates in meters.
        """
        self.bogies.position = position

    def move(self, displacement):
        """
        Move the train by a certain displacement vector.

        :param displacement (tuple or list): (dx, dy, dz) displacement in meters.
        """
        self.bogies.move(displacement)