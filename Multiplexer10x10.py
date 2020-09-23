import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 10
nx = 20
ny = 20
dpml = 2

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 5.5), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 8), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 11.5), (5, 0.5), -1000.0) #Top exit wvg
PPC.Add_Block_static((15, 14), (5, 0.5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
rod_eps = -3.0*np.ones((10, 10)) #Rod perm values
bounds = [-10,1] #Min and max perm values 
rho = PPC.Eps_to_Rho(rod_eps, bounds) #Initial Parameters
PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array


## Set up Sources and Sim #####################################################
w1 = 1.0 #Source frequency
w2 = 1.1 
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w1, 'src_1', 'hz')
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w2, 'src_2', 'hz')
PPC.Add_Probe(np.array([17,6]), np.array([17,8]), w1, 'prb_1', 'hz')
PPC.Add_Probe(np.array([17,12]), np.array([17,14]), w2, 'prb_2', 'hz')

#PPC.Viz_Sim_abs('src_1')
rho_opt = PPC.Optimize_Multiplexer(rho, bounds, 'src_1',\
                                       'src_2', 'prb_1', 'prb_2',\
                                        0.005, 1)
print(PPC.Rho_to_Eps(rho_opt, bounds))
PPC.Viz_Sim_abs_opt(rho_opt, bounds, ['src_1', 'src_2'])
