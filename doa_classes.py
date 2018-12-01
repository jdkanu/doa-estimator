import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    def plot_classes(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = []
        ys = []
        zs = []
        for doa_class in self.classes:
            xs.append(doa_class.x)
            ys.append(doa_class.y)
            zs.append(doa_class.z)
        zeros = [0]*len(self.classes)
        ax.quiver(zeros,zeros,zeros,xs,ys,zs,arrow_length_ratio=0.01)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        plt.show()

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
