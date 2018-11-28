import numpy as np

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