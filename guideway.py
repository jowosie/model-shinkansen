"""
SHINKANSEN GUIDEWAY DEFINITION
"""

import magpylib as mpl

class Guideway:
    def __init__(self, length = 50, width = 10, coil_count = 100):
        """
        Initializes guideway.
        :param length (float): Length of guideway section (meters)
        :param width (float): Width of guideway section (meters)
        :param coil_count (int): Number of guidance coils along the length
        """
        self.length = length
        self.width = width
        self.coil_count = coil_count

        self.levitation_coils = self._create_levitation_coils()
        self.guidance_coils = self.levitation_coils

    def _create_levitation_coils(self):
        """
        Creates figure eight null-flux coils for levitation and guidance.
        """
        coils = []
        coil_spacing = self.length / self.coil_count
        coil_width = 0.3 # meters
        coil_length = 0.4 # meters

        for side in [-1, 1]:
            for i in range(self.coil_count):
                x_pos = i * coil_spacing - self.length / 2
                y_pos = side * self.width / 2

                # Top loop
                loop1 = mpl.current.Loop(
                    current = 1,
                    diameter = coil_width,
                    position = (x_pos, y_pos, coil_length/2)
                )
                loop1.rotate_from_angax(angle=90, axis='y')

                # Bottom loop
                loop2 = mpl.current.Loop(
                    current = 1,
                    diameter = coil_width,
                    position = (x_pos, y_pos, -coil_length/2)
                )
                loop2.rotate_from_angax(angle=90, axis='y')

                coils.extend([loop1, loop2])

            return mpl.Collection(coils)

    def get_levitation_coils(self):
        # Returns magpylib collection
        return self.levitation_coils
