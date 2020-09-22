import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 10
nx = 20
ny = 20
dpml = 2

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 9), (5, 2), 12.0) #Add entrance wvg
PPC.Add_Block_static((15, 6), (5, 2), 12.0) #Bottom exit wvg
PPC.Add_Block_static((15, 12), (5, 2), 12.0) #Top exit wvg
rod_eps = -3.0*np.ones((10, 10)) #Rod perm values
PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array


## Set up Sources and Sim #####################################################
w1 = 1.0 #Source frequency
w2 = 1.1 
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w1, 'src_1', 'hz')
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w2, 'src_2', 'hz')
PPC.Add_Probe(np.array([17,6]), np.array([17,8]), w1, 'prb_1', 'hz')
PPC.Add_Probe(np.array([17,12]), np.array([17,14]), w2, 'prb_2', 'hz')

#PPC.Viz_Sim_abs('src_1')
rod_eps_opt = PPC.Optimize_Multiplexer(rod_eps, [-20,1], 'src_1',\
                                       'src_2', 'prb_1', 'prb_2',\
                                        0.005, 1)
PPC.Viz_Sim_abs_opt(rod_eps_opt, [-20, 1], ['src_1', 'src_2'])