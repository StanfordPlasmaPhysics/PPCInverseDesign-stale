import numpy as np
from PMMInverse import PMMI

a = 0.015
res = 50
nx = 20
ny = 20
dpml = 2
b_o = 0.007/a
b_i = 0.0065/a

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object

## Logic Gate Domain
# w = 1.0 #Source frequency
# wpmax = 0

# PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prb_t', 'hz')
# PPC.Add_Probe(np.array([9,3]), np.array([11,3]), w, 'prb_n', 'hz')
# PPC.Add_Source(np.array([9,17]), np.array([11,17]), w, 'src_c', 'hz')
# PPC.Add_Source(np.array([3,6]), np.array([3,8]), w, 'src_1', 'hz')
# PPC.Add_Source(np.array([3,12]), np.array([3,14]), w, 'src_2', 'hz')

# PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Add true wvg
# PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Add true wvg
# PPC.Add_Block_static((8.5, 0), (0.5, 5), -1000.0) #Not true wvg
# PPC.Add_Block_static((11, 0), (0.5, 5), -1000.0) #Not true wvg
# PPC.Add_Block_static((0, 5.5), (5, 0.5), -1000.0) #Bottom entrance wvg
# PPC.Add_Block_static((0, 8), (5, 0.5), -1000.0) #Bottom entrance wvg
# PPC.Add_Block_static((0, 11.5), (5, 0.5), -1000.0) #Top entrance wvg
# PPC.Add_Block_static((0, 14), (5, 0.5), -1000.0) #Top entrance wvg
# PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Continuous src wvg
# PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Continuous src wvg
# PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
# PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array

## Multiplexer Domain
# w1 = 0.25 #Source frequency
# w2 = 0.27
# wpmax = 0.35

# PPC.Add_Source(np.array([3,9]), np.array([3,11]), w1, 'src_1', 'hz')
# PPC.Add_Source(np.array([3,9]), np.array([3,11]), w2, 'src_2', 'hz')
# PPC.Add_Probe(np.array([17,6]), np.array([17,8]), w1, 'prb_1', 'hz')
# PPC.Add_Probe(np.array([17,12]), np.array([17,14]), w2, 'prb_2', 'hz')

# PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
# PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
# PPC.Add_Block_static((15, 5.5), (5, 0.5), -1000.0) #Bottom exit wvg
# PPC.Add_Block_static((15, 8), (5, 0.5), -1000.0) #Bottom exit wvg
# PPC.Add_Block_static((15, 11.5), (5, 0.5), -1000.0) #Top exit wvg
# PPC.Add_Block_static((15, 14), (5, 0.5), -1000.0) #Top exit wvg
# PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
# PPC.Rod_Array_train(0.433, (5.5, 5.5), (10, 10)) #Rod ppc array

## Waveguide Domain
w = 0.25 #Source frequency
wpmax = 0.35

# Straight
# PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'hz')
# PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prb', 'hz')
# PPC.Add_Probe(np.array([9,17]), np.array([11,17]), w, 'prbl', 'hz')

# Bent
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'ez')
PPC.Add_Probe(np.array([9,17]), np.array([11,17]), w, 'prb', 'ez')
PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prbl', 'ez')

PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(b_i, (5.5, 5.5), (10, 10), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = False) #Rod ppc array

## Read parameters and visualize ##############################################
rho = PPC.Read_Params('params/10by10bentwaveguide_ez_w025_wpmax035_gam1GHz_res75_coldstart_r8.csv')
print(PPC.Rho_to_Eps(rho = rho, plasma = True, w_src = w, wp_max = wpmax))
PPC.Params_to_Exp(rho = rho, src = 'src', plasma = True, wp_max = wpmax)
PPC.Viz_Sim_abs_opt(rho, ['src'], 'checkfields.pdf',\
                    plasma = True, wp_max = wpmax, uniform = False)