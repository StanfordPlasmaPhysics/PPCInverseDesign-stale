"""
The module composed in this file is meant to serve as a platform for
designing plasma metamaterial devices and then optimizing the plasma density
of the elements composing the metamaterial to achieve a certain functionality.
It is built atop ceviche, an autograd compliant FDFD/FDTD EM simulation tool 
(https://github.com/fancompute/ceviche).
Jesse A Rodriguez, 09/15/2020
"""

import numpy as np
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
plt.rc('font', family='tahoma')
font = 18
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
from autograd.scipy.signal import convolve as conv
from skimage.draw import disk, rectangle
import ceviche
from ceviche import fdfd_ez, jacobian, fdfd_hz
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
import collections

## Utility Functions ##########################################################
def mode_overlap(E1, E2):
    """
    Defines an overlap integral between the sim field and desired field

    Args:
        E1, E2: Matrices with solved field values
    """
    return npa.abs(npa.sum(npa.conj(E1)*E2))*1e6
###############################################################################

class PMMI:
    def __init__(self, a, res, nx, ny, dpml):
        """
        Initialize PMMI object

        Args:
            a: dimensionalized unit length (e.g. 0.01 m)
            res: pixels per a unit
            nx: number of a units in x-direction for active sim region
            ny: number of a units in y-direction for active sim region
            dpml: thickness of PML in a units
        """
        self.a = a
        self.res = res
        self.nx = nx
        self.ny = ny
        self.dpml = dpml
        self.dl = a/res #meters per pixel
        self.Npml = dpml*res #Number of pixels in PML layer
        self.Nx = nx*res #Number of pixels in x-direction
        self.Ny = ny*res #Number of pixels in y-direction
        self.epsr = np.ones((self.Nx, self.Ny)) #Initialize relative perm array
        self.design_region = np.zeros((self.Nx,self.Ny)) #Design region array
        self.train_elems = [] #Masks for trainable elements
        self.static_elems = np.ones((self.Nx, self.Ny)) #Array for static elements
                                                         #outside the training region.
        self.sources = {} #Empty dict to hold source arrays
        self.probes = {} #Empty dict to hold probe arrays

    def Design_Region(self, bot_left, extent):
        """
        Specify design region selector

        Args:
            bot_left: coords of bottom left ocrner of design region in a units
            extent: width and height of design region
        """
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        W = int(round(extent[0]*self.res))
        H = int(round(extent[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.design_region[rr, cc] = 1


    def Add_Rod(self, r, center, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            r: radius of the rod in a units
            center: x,y coords of the rod center in a units
            eps: relative permittivity
        """
        R = int(round(r*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        rr, cc = disk((X, Y), R, shape = self.epsr.shape)
        
        self.epsr[rr, cc] = eps


    def Add_Rod_train(self, r, center):
        """
        Add a single rod with radius r and rel. permittivity eps to the trainable
        element array.

        Args:
            r: radius of the rod in a units
            center: x,y coords of the rod center in a units
        """
        R = int(round(r*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        rr, cc = disk((X, Y), R, shape = self.epsr.shape)
        
        train_elem = np.zeros((self.epsr.shape))
        train_elem[rr, cc] = 1
        self.train_elems.append(train_elem)


    def Add_Block(self, bot_left, extent, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            bot_left: x,y coords of the bottom left corner in a units
            extent: width and height of the block in a units
            eps: relative permittivity of the block
        """
        H = int(round(extent[1]*self.res))
        W = int(round(extent[0]*self.res))
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.epsr[rr, cc] = eps


    def Add_Block_static(self, bot_left, extent, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to the static
        elems array.

        Args:
            bot_left: x,y coords of the bottom left corner in a units
            extent: width and height of the block in a units
            eps: relative permittivity of the block
        """
        H = int(round(extent[1]*self.res))
        W = int(round(extent[0]*self.res))
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.static_elems[rr, cc] = eps


    def Add_Source(self, xy_begin, xy_end, w, src_name, pol):
        """
        Add a source to the domain.

        Args:
            xy_begin: coords of the start of the source in a units (np.array)
            xy_end: coords of the end of the source in a units (np.array)
            w: Source frequency in c/a units
            src_name: string that serves as key for the source in the dict
            pol: string specifying the polarization of the source e.g. 'hz'
        """
        XY_beg = (np.rint(xy_begin*self.res)).astype(int)
        XY_end = (np.rint(xy_end*self.res)).astype(int)
        if XY_beg[0] == XY_end[0]:
            src_y = (np.arange(XY_beg[1],XY_end[1])).astype(int)
            src_x = XY_beg[0]*np.ones(src_y.shape, dtype=int)
        elif XY_beg[1] == XY_end[1]:
            src_x = (np.arange(XY_beg[0], XY_end[0])).astype(int)
            src_y = XY_beg[1]*np.ones(src_x.shape, dtype=int)
        else:
            raise RuntimeError("Source needs to be 1-D")

        omega = 2*np.pi*w*299792458/self.a
        src = insert_mode(omega, self.dl, src_x, src_y, self.epsr, m = 1)

        self.sources[src_name] = (src, omega, pol)
        

    def Add_Probe(self, xy_begin, xy_end, w, prb_name, pol):
        """
        Add a probe to the domain.

        Args:
            xy_begin: coords of the start of the probe in a units (np.array)
            xy_end: coords of the end of the probe in a units (np.array)
            w: Probe frequency in c/a units
            src_name: string that serves as key for the probe in the dict
            pol: string specifying the polarization of the probe e.g. 'hz'
        """
        XY_beg = (np.rint(xy_begin*self.res)).astype(int)
        XY_end = (np.rint(xy_end*self.res)).astype(int)
        if XY_beg[0] == XY_end[0]:
            prb_y = (np.arange(XY_beg[1],XY_end[1])).astype(int)
            prb_x = XY_beg[0]*np.ones(prb_y.shape, dtype=int)
        elif XY_beg[1] == XY_end[1]:
            prb_x = (np.arange(XY_beg[0], XY_end[0])).astype(int)
            prb_y = XY_beg[1]*np.ones(prb_x.shape, dtype=int)
        else:
            raise RuntimeError("Probe needs to be 1-D")

        omega = 2*np.pi*w*299792458/self.a
        prb = insert_mode(omega, self.dl, prb_x, prb_y, self.epsr, m = 1)

        self.probes[prb_name] = (prb, omega, pol)


    def Rod_Array(self, r, xy_start, rod_eps, d_x = 1, d_y = 1):
        """
        Add a 2D rectangular rod array to epsr. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            rod_eps: np array of size nrods_x by nrods_y giving the relative 
                     permittivity of each of the rods
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
        """
        for i in range(rod_eps.shape[0]):
            for j in range(rod_eps.shape[1]):
                x = xy_start[0] + i*d_x
                y = xy_start[1] + j*d_y
                rod_e = rod_eps[i,j]
                if not isinstance(rod_eps[i,j], np.float64):
                    rod_e = rod_eps[i,j]._value
                self.Add_Rod(r, (x, y), rod_e)


    def Rod_Array_train(self, r, xy_start, array_dims, d_x = 1, d_y = 1):
        """
        Add a 2D rectangular rod array to the train elems. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
        """
        for i in range(array_dims[0]):
            for j in range(array_dims[1]):
                x = xy_start[0] + i*d_x
                y = xy_start[1] + j*d_y
                self.Add_Rod_train(r, (x, y))


    def Viz_Sim_abs(self, src_names, slices=[]):
        """
        Solve and visualize an static simulation with certain sources active
        
        Args:
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if self.sources[src_names[i]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Hz.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field Magnitude', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Ez.T), cmap='magma'), ax=ax[i])
                cbar = plt.colorbar(ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        for sl in slices:
            ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.show()

        return (simulation, ax)


    def Viz_Sim_fields(self, src_names, slices=[]):
        """
        Solve and visualize a static simulation with certain sources active
        
        Args:
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if self.sources[src_names[i]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(Hz.T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='k', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(Ez.T, cmap='RdBu'), ax=ax[i])
                cbar = plt.colorbar(ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        for sl in slices:
            ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.show()

        return (simulation, ax)


    def Viz_Sim_abs_opt(self, rho, bounds, src_names, slices=[]):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            bounds: Upper and lower bounds for parameters
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        epsr_opt = self.Rho_Parameterization(rho, bounds)
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if self.sources[src_names[i]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, epsr_opt,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Hz.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field Magnitude', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, epsr_opt,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Ez.T), cmap='magma'), ax=ax[i])
                cbar = plt.colorbar(ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        for sl in slices:
            ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(epsr_opt.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.show()

        return (simulation, ax)


    def Viz_Sim_fields_opt(self, rho, bounds, src_names, slices=[]):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            bounds: Upper and lower bounds for parameters
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        epsr_opt = self.Rho_Parameterization(rho, bounds)
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if self.sources[src_names[i]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, epsr_opt,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.real(Hz).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='k', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, epsr_opt,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(Ez.T, cmap='RdBu'), ax=ax[i])
                cbar = plt.colorbar(ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                ax[i].contour(epsr_opt.T, 2, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        for sl in slices:
            ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(epsr_opt.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.show()

        return (simulation, ax)


    def Mask_Combine_Rho(self, train_epsr, elem_locs, bounds, eps_bg_des):
        """
        Utility function for combining the design region with its trainable 
        elements and the static region

        Args:
            train_epsr: An np array that is the size of the domain and contains
            zeros everywhere except the location of the trainable elements where
            their scaled epsilon values are present.
            elem_locs: similar to train_epsr but just an element selector
            bounds: Array containing the upper and lower bounds of the perm of
            the trainable elements
            eps_bg_des: Float, permittivity of the background in the design region
        """
        train = (bounds[1] - bounds[0])*train_epsr*(elem_locs!=0).astype(np.float)\
                + bounds[0]*(elem_locs!=0).astype(np.float)
        design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.float)
        bckgd = self.static_elems*(self.design_region==0).astype(np.float)

        return train + design + bckgd


    def Scale_Rho(self, rho):
        """
        Applies a non-linear activation to the optimization parameters

        Args:
            rho: Parameters being optimized
        """
        rho = rho.flatten()
        rho = npa.arctan(rho) / np.pi + 0.5
        train_epsr = np.zeros(self.train_elems[0].shape)
        elem_locations = np.zeros(self.train_elems[0].shape)
        for i in range(len(rho)):
            train_epsr += rho[i]*self.train_elems[i]
            elem_locations += self.train_elems[i]

        return train_epsr, elem_locations


    def Eps_to_Rho(self, epsr, bounds):
        """
        Returns parameters associated with array of values of epsr

        Args:
            epsr: array of relative permittivity values
        """
        return npa.tan(((epsr-bounds[0])/(bounds[1]-bounds[0])-0.5)*np.pi)

    
    def Rho_to_Eps(self, rho, bounds):
        """
        Returns permittivity values associated with a parameter matrix

        Args:
            rho: array of optimization parameters
        """
        return (bounds[1]-bounds[0])*(npa.arctan(rho)/np.pi+0.5)+bounds[0]


    def Rho_Parameterization(self, rho, bounds, eps_bg_des = 1):
        """
        Apply activation/parameterization and create a permittivity matrix

        Args:
            rho: parameters to be optimized
            bounds: upper and lower limits for the elements of rho
            eps_bg_des: background epsilon for the design/optimization region
        """
        train_epsr, elem_locs = self.Scale_Rho(rho)

        return self.Mask_Combine_Rho(train_epsr, elem_locs, bounds, eps_bg_des)


    def Optimize_Waveguide(self, Rho, bounds, src, prb, alpha, nepochs):
        """
        Optimize a waveguide PMM

        Args:
            Rho: Initial parameters
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            src: Key for the source in the sources dict.
            prb: Key for probe (slice in desired output waveguide) in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
        """
        if self.sources[src][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim = fdfd_hz(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            Ex, _, _ = sim.solve(self.sources[src][0])
            E0 = mode_overlap(Ex, self.probes[prb][0])
            
            #Define objective
            def objective(rho):
                """
                Objective function called by optimizer

                1) Takes the density distribution as input
                2) Constructs epsr
                3) Runs the simulation
                4) Returns the overlap integral between the output wg field
                and the desired mode field
                """
                rho = rho.reshape(Rho.shape)
                epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                Ex, _, _ = sim.solve(self.sources[src][0])

                return mode_overlap(Ex, self.probes[prb][0])/E0

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, _ = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape)

        elif self.sources[src][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim = fdfd_ez(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, Ez = sim.solve(self.sources[src][0])
            E0 = mode_overlap(Ez, self.probes[prb][0])
            
            #Define objective
            def objective(rho):
                """
                Objective function called by optimizer

                1) Takes the density distribution as input
                2) Constructs epsr
                3) Runs the simulation
                4) Returns the overlap integral between the output wg field
                and the desired mode field
                """
                rho = rho.reshape(Rho.shape)
                epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                _, _, Ez = sim.solve(self.sources[src][0])

                return mode_overlap(Ez, self.probes[prb][0])/E0

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, _ = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape)

        else:
            raise RuntimeError("The source polarization is not valid.")


    def Optimize_Multiplexer(self, Rho, bounds, src_1, src_2, prb_1, prb_2,\
                             alpha, nepochs):
        """
        Optimize a multiplexer PMM

        Args:
            Rho: Initial parameters
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
        """
        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            Ex1, _, _ = sim1.solve(self.sources[src_1][0])
            Ex2, _, _ = sim2.solve(self.sources[src_2][0])
            E01 = mode_overlap(Ex1, self.probes[prb_1][0])
            E02 = mode_overlap(Ex2, self.probes[prb_2][0])
            
            #Define objective
            def objective(rho):
                """
                Objective function called by optimizer

                1) Takes the density distribution as input
                2) Constructs epsr
                3) Runs the simulation
                4) Returns the overlap integral between the output wg field
                and the desired mode field
                """
                rho = rho.reshape(Rho.shape)
                epsr = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr
                sim2.eps_r = epsr

                Ex1, _, _ = sim1.solve(self.sources[src_1][0])
                Ex2, _, _ = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ex1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ex2, self.probes[prb_2][0])/E02)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, _ = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape)

        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, Ez1 = sim1.solve(self.sources[src_1][0])
            _, _, Ez2 = sim2.solve(self.sources[src_2][0])
            E01 = mode_overlap(Ez1, self.probes[prb_1][0])
            E02 = mode_overlap(Ez2, self.probes[prb_1][0])
            
            #Define objective
            def objective(rho):
                """
                Objective function called by optimizer

                1) Takes the density distribution as input
                2) Constructs epsr
                3) Runs the simulation
                4) Returns the overlap integral between the output wg field
                and the desired mode field
                """
                rho = rho.reshape(Rho.shape)
                epsr = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr
                sim2.eps_r = epsr

                _, _, Ez1 = sim1.solve(self.sources[src_1][0])
                _, _, Ez2 = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ez1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ez2, self.probes[prb_2][0])/E02)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, _ = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape)

        else:
            raise RuntimeError("The two sources must have the same polarization.")
