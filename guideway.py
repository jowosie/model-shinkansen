"""
SHINKANSEN GUIDEWAY DEFINITION
"""

import magpylib as mpl
import config


class Guideway:
    def __init__(self, length=50, width=10, coil_count=100, coil_diameter=0.05):
        """
        Initializes guideway.
        :param length (float): Length of guideway section (meters)
        :param width (float): Width of guideway section (meters)
        :param coil_count (int): Number of guidance coils along the length
        """
        self.length = length
        self.width = width
        self.coil_count = coil_count
        self.coil_diameter = coil_diameter

        self.levitation_coils = self._create_levitation_coils()
        self.guidance_coils = self.levitation_coils
        self.propulsion_coils = self._create_propulsion_coils()

    def _create_levitation_coils(self):
        """
        Creates figure eight null-flux coils for levitation and guidance using Loop objects.
        """
        coils = []
        coil_spacing = self.length / self.coil_count
        coil_width = 0.3  # meters
        coil_length = 0.4  # meters

        for side in [-1, 1]:
            for i in range(self.coil_count):
                x_pos = i * coil_spacing - self.length / 2
                y_pos = side * self.width / 2

                # Top loop
                loop1 = mpl.current.Loop(
                    current=1,
                    diameter=coil_width,
                    position=(x_pos, y_pos, coil_length / 2)
                )
                loop1.rotate_from_angax(angle=90, axis='y')
                loop1.meshing = (5, 5)

                # Bottom loop
                loop2 = mpl.current.Loop(
                    current=1,
                    diameter=coil_width,
                    position=(x_pos, y_pos, -coil_length / 2)
                )
                loop2.rotate_from_angax(angle=90, axis='y')
                loop2.meshing = (5, 5)

                coils.extend([loop1, loop2])

        return mpl.Collection(coils)

    def get_levitation_coils(self):
        # Returns magpylib collection
        return self.levitation_coils

    def _create_propulsion_coils(self):
        """
        Creates actively powered LSM coils for propulsion using Line objects.
        """
        prop_coils = []
        num_coils_per_side = int(self.length * 2)
        coil_side_length = self.coil_diameter

        # Use same spacing as magnet spacing
        magnet_width_spacing = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]

        for side in [-1, 1]:
            for i in range(num_coils_per_side):
                x_pos = (i - num_coils_per_side / 2) * 0.5
                y_pos = side * (magnet_width_spacing / 2 - 0.2)
                z_pos = 0

                # Define the four corners of a square path for the current
                vertices = [
                    (x_pos - coil_side_length / 2, y_pos, z_pos - coil_side_length / 2),
                    (x_pos + coil_side_length / 2, y_pos, z_pos - coil_side_length / 2),
                    (x_pos + coil_side_length / 2, y_pos, z_pos + coil_side_length / 2),
                    (x_pos - coil_side_length / 2, y_pos, z_pos + coil_side_length / 2),
                    (x_pos - coil_side_length / 2, y_pos, z_pos - coil_side_length / 2)
                ]

                coil = mpl.current.Line(
                    current=0,
                    vertices=vertices
                )

                coil.meshing = 20
                prop_coils.append(coil)

        return mpl.Collection(prop_coils)

    def get_propulsion_coils(self):
        return self.propulsion_coils