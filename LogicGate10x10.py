import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 50
nx = 20
ny = 20
dpml = 2

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Add true wvg
PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Add true wvg
PPC.Add_Block_static((8.5, 0), (0.5, 5), -1000.0) #Not true wvg
PPC.Add_Block_static((11, 0), (0.5, 5), -1000.0) #Not true wvg
PPC.Add_Block_static((0, 5.5), (5, 0.5), -1000.0) #Bottom entrance wvg
PPC.Add_Block_static((0, 8), (5, 0.5), -1000.0) #Bottom entrance wvg
PPC.Add_Block_static((0, 11.5), (5, 0.5), -1000.0) #Top entrance wvg
PPC.Add_Block_static((0, 14), (5, 0.5), -1000.0) #Top entrance wvg
PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Continuous src wvg
PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Continuous src wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 1.0 #Source frequency

PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prb_t', 'hz')
PPC.Add_Probe(np.array([9,3]), np.array([11,3]), w, 'prb_n', 'hz')
PPC.Add_Source(np.array([9,17]), np.array([11,17]), w, 'src_c', 'hz')
PPC.Add_Source(np.array([3,6]), np.array([3,8]), w, 'src_1', 'hz')
PPC.Add_Source(np.array([3,12]), np.array([3,14]), w, 'src_2', 'hz')

rod_eps = 0.75*np.ones((10, 10)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w) #Initial Parameters

rho_opt, obj = PPC.Optimize_Logic_Gate(rho, 'src_1', 'src_2', 'src_c', 'prb_n',\
                                        'prb_t', 0.002, 1400, 'and', plasma = True)

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, 'params/10by10logic_and_hz_w1_wp_pen.csv')
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, [['src_c'], ['src_c', 'src_1'], ['src_c','src_2'],\
        ['src_c', 'src_1', 'src_2']], 'plots/10by10Logic_And_Hz_w1_wp_pen.pdf',\
        plasma = True, mult = True)
PPC.Viz_Obj(obj, 'plots/10by10Logic_And_Hz_w1_wp_obj_pen.pdf')