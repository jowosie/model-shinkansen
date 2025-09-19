"""
SHINKANSEN GUIDEWAY DEFINITION
"""

import magpylib as mpl
import config


class Guideway:
    def __init__(self, length, width=10, coil_count=100, coil_diameter=0.05):
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

        # Space coils according to pole pitch for proper LSM operation
        magnet_pole_pitch = config.SCMAGLEV_SYSTEM["lsm_pole_pitch"]

        # Number of coils based on track length and pole pitch
        # Three coils per wavelength for three-phase system
        wavelength = 2 * magnet_pole_pitch
        coils_per_wavelength = 3
        num_wavelengths = int(self.length / wavelength)
        num_coils_per_side = num_wavelengths * coils_per_wavelength

        coil_spacing = self.length / num_coils_per_side if num_coils_per_side > 0 else magnet_pole_pitch
        coil_side_length = 0.3  # Size of each coil

        for side in [-1, 1]:
            for i in range(num_coils_per_side):
                # Position along the track
                x_pos = (i - num_coils_per_side / 2) * coil_spacing

                # Lateral position - closer to centerline for better coupling
                y_pos = side * (self.width / 4)  # Place at 1/4 of track width
                z_pos = 0

                # Define the four corners of a rectangular coil
                half_length = coil_side_length / 2
                half_width = 0.1  # Narrower in y-direction

                vertices = [
                    (x_pos - half_length, y_pos - half_width, z_pos - half_length),
                    (x_pos + half_length, y_pos - half_width, z_pos - half_length),
                    (x_pos + half_length, y_pos + half_width, z_pos - half_length),
                    (x_pos - half_length, y_pos + half_width, z_pos - half_length),
                    (x_pos - half_length, y_pos - half_width, z_pos - half_length),
                    (x_pos - half_length, y_pos - half_width, z_pos + half_length),
                    (x_pos + half_length, y_pos - half_width, z_pos + half_length),
                    (x_pos + half_length, y_pos + half_width, z_pos + half_length),
                    (x_pos - half_length, y_pos + half_width, z_pos + half_length),
                    (x_pos - half_length, y_pos - half_width, z_pos + half_length)
                ]

                coil = mpl.current.Line(
                    current=0,
                    vertices=vertices
                )

                coil.meshing = 10
                prop_coils.append(coil)

        return mpl.Collection(prop_coils)

    def get_propulsion_coils(self):
        return self.propulsion_coils