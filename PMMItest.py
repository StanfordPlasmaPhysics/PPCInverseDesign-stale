import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 35
nx = 20
ny = 20
dpml = 2

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block(5, 2, 0, 9, 12.0) #Add entrance wvg
PPC.Add_Block(5, 2, 15, 6, 12.0) #Bottom exit wvg
PPC.Add_Block(5, 2, 15, 12, 12.0) #Top exit wvg
rod_eps = 5.0*np.ones((10,10)) #Rod perm values
PPC.Rod_Array(0.433, 5.5, 5.5, 10, 10, rod_eps) #Rod ppc array


## Set up Source and Sim ######################################################
w = 1.0 #Source frequency
PPC.Add_Source(np.array([3,8]), np.array([3,12]), w, 'src_1', 'ez')
sim, ax = PPC.Viz_Sim_abs('src_1')