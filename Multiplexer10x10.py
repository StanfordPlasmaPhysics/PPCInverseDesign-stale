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
PPC.Add_Block_static((15, 5.5), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 8), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 11.5), (5, 0.5), -1000.0) #Top exit wvg
PPC.Add_Block_static((15, 14), (5, 0.5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array


## Set up Sources and Sim #####################################################
w1 = 1.0 #Source frequency
w2 = 1.1 
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w1, 'src_1', 'hz')
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w2, 'src_2', 'hz')
PPC.Add_Probe(np.array([17,6]), np.array([17,8]), w1, 'prb_1', 'hz')
PPC.Add_Probe(np.array([17,12]), np.array([17,14]), w2, 'prb_2', 'hz')

rod_eps = 0.75*np.ones((10, 10)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w1) #Initial Parameters

rho_opt, obj = PPC.Optimize_Multiplexer_Penalize(rho, 'src_1', 'src_2', 'prb_1',\
                                            'prb_2', 0.002, 1200, plasma = True)

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, 'params/10by10multiplexer_hz_w1_w1-1_wp_pen.csv')
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w1))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, ['src_1', 'src_2'],\
                    'plots/Multiplexer_Hz_w1_w1-1_wp_pen.pdf', plasma = True)
PPC.Viz_Obj(obj, 'plots/Multiplexer10by10_Hz_w1_w1-1_wp_obj_pen.pdf')
