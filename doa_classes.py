import numpy as np

class DoaClasses():
    def __init__(self, doa_grid_resolution = np.pi/18):
        self.classes = generate_direction_classes(doa_grid_resolution)

    def xyz_at_index(self, index):
        assert index < len(self.classes)
        doa_class = self.classes[index]
        azimuth = doa_class["azimuth"]
        inclination = (np.pi/2) - doa_class["elevation"]
        
        x = np.sin(inclination)*np.cos(azimuth)
        y = np.sin(inclination)*np.sin(azimuth)
        z = np.cos(inclination)
        return (x, y, z)

# Produces list of {elevation: ..., azimuth: ...} with values in radians.
# Input is resolution of grid in radians
def generate_direction_classes(resolution):
    direction_classes = []
    num_elevations = int(np.pi//resolution)
    for i in range(num_elevations):
        elevation = (-np.pi/2) + np.pi*i/num_elevations
        num_azimuths = int(2*np.pi*np.cos(elevation)//resolution)
        for j in range(num_azimuths):
            azimuth = -np.pi + 2*np.pi*j/(num_azimuths + 1)
            direction_classes.append({"elevation": elevation, "azimuth": azimuth})

    return direction_classes