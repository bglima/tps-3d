"""
References:
https://khanhha.github.io/posts/Thin-Plate-Splines-Warping/
https://profs.etsmtl.ca/hlombaert/thinplates/
https://www.cse.wustl.edu/~taoju/cse554/lectures/lect07_Deformation2.pdf
https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py
"""
#%%
import numpy as np
import pickle
np.set_printoptions(precision=4, suppress=True)

class TPS3D:
    def __init__(self):
        pass

    def distance(self, p1, p2):
        """
        Distance calculation between p1 and P2.
        TPS uses radial basis distance of form r^2 * log (r).
        x1 and x2 are 3d points.
        """
        diff = np.array(p1) - np.array(p2)
        norm = np.linalg.norm(diff)
        return norm * norm * np.log(norm) if norm > 0 else 0

    def fit(self, c_points, t_points):
        """
        Parameters are control and target points of type numpy_array, both with dimension [m x 3].
        """
        assert c_points.shape[0] == t_points.shape[0], "Control and target array points must have same dimension."
        m = c_points.shape[0] 
        self.c_points = c_points
        self.t_points = t_points

        """
        K matrix contains the spatial relation between control points, calculating their distance 2-by-2.
        P matrix contains the control_points in homogeneous form (1, c_x, c_y, c_z)
        """
        self.K = np.zeros((m, m))
        self.P = np.zeros((m, 4))
        
        for i in range(m):
            for j in range(m):
                self.K[i, j] = self.distance(c_points[i], c_points[j])
            self.P[i, 0] = 1
            self.P[i, 1:] = c_points[i]

        """
        L matrix aggregate information about K, P and P.T blocks
        """
        self.L = np.zeros((m+4, m+4))
        self.L[:m, :m] = self.K
        self.L[:m, m:] = self.P
        self.L[m:, :m] = self.P.T

        # Building the right hand side B = [v 0].T. 
        self.B = np.zeros((m+4, 3))
        self.B[:m, :] = t_points

        # Solve for TPS parameters
        # Find vector  = [w a]^T
        self.tps_params = np.linalg.solve(self.L, self.B)

    def transform(self, p):
        """
        Apply a tps warp into a 3d numpy array point. Returns the point after transformation.
        """
        output_point = np.zeros((3, 1))
        for i in range(3):
            # Calculate affine term
            a1 = self.tps_params[-4, i]
            ax = self.tps_params[-3, i]
            ay = self.tps_params[-2, i]
            az = self.tps_params[-1, i]
            
            affine = a1 + ax*p[0] + ay*p[1] + az*p[2]

            # Calculate non-rigid deformation.
            # Sum up contribuitions for all kernels, given the kernel weights
            nonrigid = 0
            for j in range(self.c_points.shape[0]):
                nonrigid += self.tps_params[j, i] * self.distance(self.c_points[j], p)

            output_point[i] = affine + nonrigid
        return output_point

    def assessModel(self):
        # check errors between target and transformed points
        total_error = 0
        for (index, input_point) in enumerate(self.c_points):
            output_point = self.transform(input_point).squeeze()
            target_point = self.t_points[index].squeeze()
            error_vector = np.around(target_point - output_point, decimals=5)
            total_error += np.linalg.norm(error_vector)
        return total_error

    def save(self, filename='tps_model.pkl'):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)

    def load(self, filename='tps_model.pkl'):
        with open(filename, 'rb') as input_file:
            loaded_model = pickle.load(input_file)
            self.__class__ = loaded_model.__class__
            self.__dict__ = loaded_model.__dict__

if __name__ == '__main__':
    """
    Testing the TPS model with sample points.
    """
    # Control points
    xs = [-411.34, -276.03, -98.33, 40.22, 136.59, -104.59, -251.04, -318.4, -316.81, -212.5, -40.54, 
        67.61, 165.82, -72.47, -285.28, -368.06]
    ys = [-143.36, -153.81, -133.12, -137.78, -113.75, -103.84, -94.38, -55.27, 53.21, 22.24, 12.59, 
        -18.12, -19.11, 32.67, 26.69, 38.13]
    zs = [-371.41, -391.57, -364.37, -290.95, -158.73, -221.63, -187.93, -398.02, -406.38, -453.3, -390.13, 
        -266.99, -134.02, -185.94, -189.68, -373.12]
    c_points = np.vstack([xs, ys, zs]).T

    # Target points
    xt = [0.32, 0.35, 0.32, 0.2, 0.12, 0.15, 0.12, 0.2, 0.27, 0.29, 0.27, 0.2, 0.16, 0.18, 0.16, 0.2]
    yt = [-0.15, 0, 0.15, 0.12, 0.08, 0, -0.08, -0.12, -0.14, 0, 0.14, 0.12, 0.08, 0, -0.08, -0.12]
    zt = [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    t_points = np.vstack([xt, yt, zt]).T

    # Instantiating TPS model
    tps_model = TPS3D()
    tps_model.fit(c_points, t_points)

    p = np.array([-411.34, -143.36, -371.41])
    print("Output for {} is {}: ".format(p, tps_model.transform(p).T.squeeze() ) )

    total_error = tps_model.assessModel()
    print("Total absolute error for control and target points is: {}".format(total_error))

#%%