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

    def create_bogies(self):
        """Creates the magnet arrangement for the train's bogies"""
        all_magnets = []
        bogie_length = 3 # meters
        magnet_spacing_length = bogie_length / (self.magnets_per_bogie / 2)
        magnet_width_spacing = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]

        for i in range(self.num_bogies):
            for j in range(self.magnets_per_bogie):
                # Sets alternating polarities for magnets
                polarity = 1 if j % 2 == 0 else -1

                x_pos = (i * 15)
                y_pos = -magnet_width_spacing/2 + (j % 2) * magnet_width_spacing
                z_pos = (j // 2) * magnet_spacing_length - (bogie_length / 2)

                magnet = mpl.magnet.Cuboid(
                    magnetization = (0,0, polarity * self.magnet_strength),
                    dimension = (0.5, 0.2, 0.4), # meters (length, width, height)
                    position = (x_pos, y_pos, z_pos)
                )
                all_magnets.append(magnet)

                magnet.meshing = (5,5,5)

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