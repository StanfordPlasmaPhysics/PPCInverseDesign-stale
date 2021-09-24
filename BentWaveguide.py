import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 80
nx = 20
ny = 20
dpml = 2
b_o = 0.007/a
b_i = 0.0065/a

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(b_i, (5.5, 5.5), (10, 10), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = False) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 0.25 #Source frequency
wpmax = 0.35
gamma = PPC.gamma(1e9)

PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'ez')
PPC.Add_Probe(np.array([9,17]), np.array([11,17]), w, 'prb', 'ez')
PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prbl', 'ez')

rod_eps = 0.999*np.ones((10, 10)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params('params/10by10bentwaveguide_ez_w025_wpmax035_gam1GHz_res80_coldstart_r4.csv')

rho_opt, obj = PPC.Optimize_Waveguide_Penalize(rho, 'src', 'prb', 'prbl',\
               0.001, 250, plasma = True, wp_max = wpmax, gamma = gamma, uniform = False,\
               param_evolution = True)

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, 'params/10by10bentwaveguide_ez_w025_wpmax035_gam1GHz_res80_coldstart_r1.csv')
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, ['src'], 'plots/BentWaveguide_Ez_w025_wpmax035_gam1GHz_res80_coldstart_r1.pdf', plasma = True)
PPC.Viz_Obj(obj, 'plots/BentWaveguide10by10_Ez_w025_wpmax035_gam1GHz_res80_coldstart_obj_r1.pdf')
