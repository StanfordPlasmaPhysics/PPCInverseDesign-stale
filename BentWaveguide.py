import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 50
nx = 20
ny = 20
dpml = 2

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
rod_eps = 0.0*np.ones((10, 10)) #Rod perm values
bounds = [-10,1] #Min and max perm values 
rho = PPC.Eps_to_Rho(rod_eps, bounds) #Initial Parameters
PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 1.0 #Source frequency
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'hz')
PPC.Add_Probe(np.array([9,17]), np.array([11,17]), w, 'prb', 'hz')

rho_opt = PPC.Optimize_Waveguide(rho, bounds, 'src', 'prb', 0.005, 300)
PPC.Save_Params(rho_opt, 'params/10by10bentwaveguide.csv')
print(PPC.Rho_to_Eps(rho_opt, bounds))
PPC.Params_to_Exp(rho_opt, bounds, 'src', 0)
PPC.Viz_Sim_abs_opt(rho_opt, bounds, ['src'], 'plots/BentWaveguide_Hz.pdf')
