from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

"""https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html"""


f = KalmanFilter (dim_x=2, dim_z=1) # (dim_x=6, dim_z=3)

f.x = np.array([[2.],    # position
                [0.]])   # velocity

# f.x = np.array([[2., 2., 2.],    # position
#                 [0., 0., 0.]])   # velocity

f.F = np.array([[1.,1.],
                [0.,1.]])

# f.F = np.array([[1.,1.],
#                 [0.,1.]])     it stays the same

# model the human controlling the body as a large disturbance?

f.H = np.array([[1.,0.]])

f.P *= 1000.

f.R = 5

# f.R =np.array([[5., 5., 5.]])

f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

# f.Q = Q_discrete_white_noise(dim=6, dt=0.1, var=0.13)

sim_sensor = [1., 1., 1., 1., 1., 2., 3., 4., 5., 5., 5., 5., 3., 1., 1., 1.]

# z = get_sensor_reading()
for z in sim_sensor:
    f.predict()
    f.update(z)
    pos = f.x

# do_something_with_estimate (f.x)
