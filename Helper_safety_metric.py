import numpy as np
from Helper_Sys_setup import Scenario






def distance_to_obstacle(x, sc: Scenario):
    pos = x[:2]
    return np.linalg.norm(pos - sc.obstacle_center) - sc.obstacle_radius  # clearance to obstacle boundary

