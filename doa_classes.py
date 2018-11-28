import numpy as np

class DoaClasses():
    def __init__(self, doa_grid_resolution = np.pi/18):
        self.classes = self.generate_direction_classes(doa_grid_resolution) 

    def xyz_at_index(self, index):
        assert index < len(self.classes)
        doa_class = self.classes[index]
        return doa_class.get_xyz_vector()

    def index_for_xyz(self, xyz):
        max_dot_product = -1
        class_index = -1
        for i, doa_class in enumerate(self.classes):
            dp = np.dot(xyz, doa_class.get_xyz_vector())
            if dp > max_dot_product:
                max_dot_product = dp
                class_index = i
        assert class_index > -1
        return class_index

    def generate_direction_classes(self, resolution):
        direction_classes = []
        num_elevations = int(np.pi//resolution)
        for i in range(num_elevations):
            elevation = (-np.pi/2) + np.pi*i/num_elevations
            num_azimuths = int(2*np.pi*np.cos(elevation)//resolution)
            for j in range(num_azimuths):
                azimuth = -np.pi + 2*np.pi*j/(num_azimuths + 1)
                direction_classes.append(DoaClass(elevation, azimuth))

        return direction_classes

class DoaClass():
    def __init__(self, elevation, azimuth):
        self.elevation = elevation
        self.azimuth = azimuth
        self.inclination = (np.pi/2) - self.elevation
        self.x = np.sin(self.inclination)*np.cos(self.azimuth)
        self.y = np.sin(self.inclination)*np.sin(self.azimuth)
        self.z = np.cos(self.inclination)

    def get_xyz_vector(self):
        return np.array([self.x, self.y, self.z])
