import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 30
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
w = 2.0 #Source frequency

PPC.Add_Source(np.array([3,6]), np.array([3,8]), w, 'src_1', 'ez')
PPC.Add_Source(np.array([3,12]), np.array([3,14]), w, 'src_2', 'ez')
PPC.Add_Source(np.array([9,17]), np.array([11,17]), w, 'src_c', 'ez')

rho_opt = PPC.Read_Params('params/10by10logic_or_ez_w2_wp_pen_4_3_1_1p5_1_1p5_1_1.csv') #Optimal Parameters

## Perturb and Visualize #####################################################
p = 0

print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True)
for i in range(10):
    PPC.Viz_Sim_abs_opt(rho_opt, [['src_c'], ['src_c', 'src_1'], ['src_c','src_2'],\
        ['src_c', 'src_1', 'src_2']], 'plots/10by10Logic_Or_Ez_w2_wp_pen_Pert_'\
        +str(p)+'_r'+str(i)+'.pdf', plasma = True, mult = True, perturb = p)