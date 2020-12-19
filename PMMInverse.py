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
c = 299792458

def mode_overlap(E1, E2):
    """
    Defines an overlap integral between the sim field and desired field

    Args:
        E1, E2: Matrices with solved field values
    """
    return npa.abs(npa.sum(npa.conj(E1)*E2))*1e6

def field_mag_int(E, mask):
    """
    Computes the integral of |E|^2 over a given mask

    Args:
        E: matrix containing field values
        mask: matrix where all entries are zero except the entries over which
        you would like to integrate
    """
    return npa.abs(npa.sum(npa.multiply(npa.multiply(npa.conj(E), E), mask)))*1e6
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

        omega = 2*np.pi*w*c/self.a
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

        omega = 2*np.pi*w*c/self.a
        prb = insert_mode(omega, self.dl, prb_x, prb_y, self.epsr, m = 1)
        mask = np.zeros((self.Nx, self.Ny))
        for i in range(len(prb_x)):
            mask[prb_x[i], prb_y[i]] = 1

        self.probes[prb_name] = (prb, omega, pol, mask)


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


    def Viz_Sim_abs(self, src_names, savepath):
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
                #ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Ez.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                #ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
                
        cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.savefig(savepath)
        plt.show()

        return (simulation, ax)


    def Viz_Sim_fields(self, src_names, savepath):
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
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(Ez.T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
                
        cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = bounds[0], vmax = bounds[1]), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.savefig(savepath)
        plt.show()

        return (simulation, ax)


    def Viz_Sim_abs_opt(self, rho, src_names, savepath, bounds = [], plasma = False,\
                        show = True, mult = False, wp_max = 0):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            src_names: list of strings that indicate which sources you'd like to simulate
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            show: bool determining if the plot is shown
            mult: bool determining if multiple sources are activated at once
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if mult:
                w_src = self.sources[src_names[i][0]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i][0]][2]
                w = self.sources[src_names[i][0]][1]
                src = self.sources[src_names[i][0]][0]
                for j in range(len(src_names[i])-1):
                    src = src + self.sources[src_names[i][j+1]][0]
            else:
                w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i]][2]
                w = self.sources[src_names[i]][1]
                src = self.sources[src_names[i]][0]
            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds)
            if pol == 'hz':
                simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(src)
                cbar = plt.colorbar(ax[i].imshow(np.abs(Hz.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field Magnitude', fontsize=font)
                #ax[i].contour(epsr_opt.T, 1, colors='w', alpha=0.5)
            elif pol == 'ez':
                simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(src)
                cbar = plt.colorbar(ax[i].imshow(np.abs(Ez.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                #ax[i].contour(epsr_opt.T, 1, colors='w', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        #for sl in slices:
        #    ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(epsr_opt.T, cmap='RdGy',\
                            vmin = np.min(self.Rho_to_Eps(rho, bounds = bounds, plasma = plasma, w_src = w_src)),\
                            vmax = np.max(epsr_opt)), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.savefig(savepath)
        if show:
            plt.show()

        return (simulation, ax)


    def Viz_Sim_fields_opt(self, rho, src_names, savepath, bounds = [], plasma = False,\
                           show = True, mult = False, wp_max = 0):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            src_names: list of strings that indicate which sources you'd like to simulate
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            show: bool determining if the plot is shown
            mult: bool determining if multiple sources are activated at once
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if mult:
                w_src = self.sources[src_names[i][0]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i][0]][2]
                w = self.sources[src_names[i][0]][1]
                src = self.sources[src_names[i][0]][0]
                for j in range(len(src_names[i])-1):
                    src = src + self.sources[src_names[i][j+1]][0]
            else:
                w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i]][2]
                w = self.sources[src_names[i]][1]
                src = self.sources[src_names[i]][0]
            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds)
            if pol == 'hz':
                simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(src)
                cbar = plt.colorbar(ax[i].imshow(np.real(Hz).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field', fontsize=font)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif pol == 'ez':
                simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(src)
                cbar = plt.colorbar(ax[i].imshow(Ez.T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
        #for sl in slices:
        #    ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        cbar = plt.colorbar(ax[len(src_names)].imshow(epsr_opt.T, cmap='RdGy',\
                            vmin = np.min(self.Rho_to_Eps(rho, bounds = bounds, plasma = plasma, w_src = w_src)),\
                            vmax = np.max(epsr_opt)), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        plt.savefig(savepath)
        if show:
            plt.show()

        return (simulation, ax)


    def Viz_Obj(self, obj, savepath, show = True):
        """
        Plot evolution of objective function throughout the training

        Args:
            obj: Objective array returned by the Adam Optimizer
            savepath: Save path (string)
        """
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(obj, linewidth=3.0)
        ax.set_xlabel('Training Epoch', fontsize = font)
        ax.set_ylabel('Objective', fontsize = font)
        plt.savefig(savepath)
        if show:
            plt.show()

        return ax


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


    def Mask_Combine_Rho_wp(self, train_epsr, elem_locs, eps_bg_des):
        """
        Utility function for combining the design region with its trainable 
        elements and the static region when mapping directly from wp values

        Args:
            train_epsr: An np array that is the size of the domain and contains
            zeros everywhere except the location of the trainable elements where
            their scaled epsilon values are present.
            elem_locs: similar to train_epsr but just an element selector
            w_src: Source frequency
            eps_bg_des: Float, permittivity of the background in the design region
        """
        train = (train_epsr)*(elem_locs!=0).astype(np.float)
        design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.float)
        bckgd = self.static_elems*(self.design_region==0).astype(np.float)

        return train + design + bckgd


    def Scale_Rho(self, rho):
        """
        Scales parameters to the space (0,1). NOTE: Runs into vanishing gradient
        issues near the edges of the allowed values for eps

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


    def Scale_Rho_wp(self, rho, w_src, wp_max = 0):
        """
        Basically applies an absolute value to the parameters so negative plasma
        frequencies aren't fed to the mask combine rho function

        Args:
            rho: Parameters being optimized
        """
        rho = rho.flatten()
        if wp_max > 0:
            rho = (wp_max/1.5)*npa.arctan(rho/(wp_max/7.5))
        rho = npa.subtract(1, npa.divide(npa.power(npa.abs(rho), 2), w_src**2))
        train_epsr = np.zeros(self.train_elems[0].shape)
        elem_locations = np.zeros(self.train_elems[0].shape)
        for i in range(len(rho)):
            train_epsr += rho[i]*self.train_elems[i]
            elem_locations += self.train_elems[i]
        
        return train_epsr, elem_locations


    def Eps_to_Rho(self, epsr, bounds=[], plasma = False, w_src = 1):
        """
        Returns parameters associated with array of values of epsr

        Args:
            epsr: array of relative permittivity values
            bounds: Max and min perm values if directly training eps
            plasma: boolean determining whether or not you're directly
                    optimizing wp.
            w_src: source frequency
        """
        if plasma:
            return self.Eps_to_Rho_wp(epsr, w_src)
        else:
            return npa.tan(((epsr-bounds[0])/(bounds[1]-bounds[0])-0.5)*np.pi)

    
    def Rho_to_Eps(self, rho, bounds=[], plasma = False, w_src = 1):
        """
        Returns permittivity values associated with a parameter matrix

        Args:
            rho: array of optimization parameters
            bounds: Min and max eps values for training if directly optimizing
                    eps
            plasma: bool specifying if params map to wp
            w_src: source frequency
        """
        if plasma:
            return self.Rho_to_Eps_wp(rho, w_src)
        else:
            return (bounds[1]-bounds[0])*(npa.arctan(rho)/np.pi+0.5)+bounds[0]


    def Eps_to_Rho_wp(self, epsr, w_src):
        """
        Returns parameters associated with array of values of epsr

        Args:
            epsr: array of relative permittivity values
            w_src: source frequency
        """
        if np.max(epsr) >= 1:
            raise RuntimeError("One or more of the permittivity values is invalid.\
                    Choose relative permittivity values less than 1.")
        return (w_src**2*(1-epsr))**0.5

    
    def Rho_to_Eps_wp(self, rho, w_src):
        """
        Returns permittivity values associated with a parameter matrix

        Args:
            rho: array of optimization parameters
            w_src: source frequency
        """
        return 1-(npa.abs(rho))**2/w_src**2


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


    def Rho_Parameterization_wp(self, rho, w_src, wp_max = 0, eps_bg_des = 1):
        """
        Apply scaling/parameterization and create a permittivity matrix when
        mapping plasma frequency to permittivity

        Args:
            rho: parameters to be optimized
            w_src: Source frequency, non-dimensionalized
            eps_bg_des: background epsilon for the design/optimization region
        """
        train_epsr, elem_locs = self.Scale_Rho_wp(rho, w_src, wp_max)

        return self.Mask_Combine_Rho_wp(train_epsr, elem_locs, eps_bg_des)


    def Optimize_Waveguide(self, Rho, src, prb, alpha, nepochs, bounds = [],\
            plasma = False, wp_max = 0):
        """
        Optimize a waveguide PMM

        Args:
            Rho: Initial parameters
            src: Key for the source in the sources dict.
            prb: Key for probe (slice in desired output waveguide) in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
        """
        if self.sources[src][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max)
            else:
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                Ex, _, _ = sim.solve(self.sources[src][0])

                return mode_overlap(Ex, self.probes[prb][0])/E0

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        elif self.sources[src][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max)
            else:
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                _, _, Ez = sim.solve(self.sources[src][0])

                return mode_overlap(Ez, self.probes[prb][0])/E0

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        else:
            raise RuntimeError("The source polarization is not valid.")


    def Optimize_Waveguide_Penalize(self, Rho, src, prb, prbl, alpha, nepochs,\
            bounds = [], plasma = False, wp_max = 0):
        """
        Optimize a waveguide PMM

        Args:
            Rho: Initial parameters
            src: Key for the source in the sources dict.
            prb: Key for probe (slice in desired output waveguide) in the probes dict.
            prbl: Key for probe at slice that you want to penalize.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
        """
        if self.sources[src][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim = fdfd_hz(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            Ex, _, _ = sim.solve(self.sources[src][0])
            E0 = mode_overlap(Ex, self.probes[prb][0])
            E0l = field_mag_int(Ex, self.probes[prbl][3])
            
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                Ex, _, _ = sim.solve(self.sources[src][0])

                return mode_overlap(Ex, self.probes[prb][0])/E0-\
                       field_mag_int(Ex, self.probes[prbl][3])/E0l

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        elif self.sources[src][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init = self.Rho_Parameterization(Rho, bounds)
            sim = fdfd_ez(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, Ez = sim.solve(self.sources[src][0])
            E0 = mode_overlap(Ez, self.probes[prb][0])
            E0l = field_mag_int(Ez, self.probes[prbl][3])
            
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                _, _, Ez = sim.solve(self.sources[src][0])

                return mode_overlap(Ez, self.probes[prb][0])/E0-\
                       field_mag_int(Ez, self.probes[prbl][3])/E0l

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        else:
            raise RuntimeError("The source polarization is not valid.")


    def Optimize_Multiplexer(self, Rho, src_1, src_2, prb_1, prb_2,\
                             alpha, nepochs, bounds = [], plasma = False,\
                             wp_max = 0):
        """
        Optimize a multiplexer PMM

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
        """
        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init1 = self.Rho_Parameterization(Rho, bounds)
                epsr_init2 = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init2,\
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
                if plasma:
                    epsr1 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                    epsr2 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr1 = self.Rho_Parameterization(rho, bounds)
                    epsr2 = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr1
                sim2.eps_r = epsr2

                Ex1, _, _ = sim1.solve(self.sources[src_1][0])
                Ex2, _, _ = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ex1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ex2, self.probes[prb_2][0])/E02)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init1 = self.Rho_Parameterization(Rho, bounds)
                epsr_init2 = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            _, _, Ez1 = sim1.solve(self.sources[src_1][0])
            _, _, Ez2 = sim2.solve(self.sources[src_2][0])
            E01 = mode_overlap(Ez1, self.probes[prb_1][0])
            E02 = mode_overlap(Ez2, self.probes[prb_2][0])
            
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
                if plasma:
                    epsr1 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                    epsr2 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr1 = self.Rho_Parameterization(rho, bounds)
                    epsr2 = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr1
                sim2.eps_r = epsr2

                _, _, Ez1 = sim1.solve(self.sources[src_1][0])
                _, _, Ez2 = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ez1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ez2, self.probes[prb_2][0])/E02)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        else:
            raise RuntimeError("The two sources must have the same polarization.")


    def Optimize_Multiplexer_Penalize(self, Rho, src_1, src_2, prb_1, prb_2,\
                             alpha, nepochs, bounds = [], plasma = False,\
                             wp_max = 0):
        """
        Optimize a multiplexer PMM with leak into opposite gate penalized.

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
        """
        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init1 = self.Rho_Parameterization(Rho, bounds)
                epsr_init2 = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            Ex1, _, _ = sim1.solve(self.sources[src_1][0])
            Ex2, _, _ = sim2.solve(self.sources[src_2][0])
            E01 = mode_overlap(Ex1, self.probes[prb_1][0])
            E02 = mode_overlap(Ex2, self.probes[prb_2][0])
            E01l = field_mag_int(Ex1, self.probes[prb_2][3])
            E02l = field_mag_int(Ex2, self.probes[prb_1][3])
            
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
                if plasma:
                    epsr1 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                    epsr2 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr1 = self.Rho_Parameterization(rho, bounds)
                    epsr2 = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr1
                sim2.eps_r = epsr2

                Ex1, _, _ = sim1.solve(self.sources[src_1][0])
                Ex2, _, _ = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ex1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ex2, self.probes[prb_2][0])/E02)-\
                       (field_mag_int(Ex1, self.probes[prb_2][3])/E01l)*\
                       (field_mag_int(Ex2, self.probes[prb_1][3])/E02l)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init1 = self.Rho_Parameterization(Rho, bounds)
                epsr_init2 = self.Rho_Parameterization(Rho, bounds)
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            _, _, Ez1 = sim1.solve(self.sources[src_1][0])
            _, _, Ez2 = sim2.solve(self.sources[src_2][0])
            E01 = mode_overlap(Ez1, self.probes[prb_1][0])
            E02 = mode_overlap(Ez2, self.probes[prb_2][0])
            E01l = field_mag_int(Ez1, self.probes[prb_2][3])
            E02l = field_mag_int(Ez2, self.probes[prb_1][3])
            
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
                if plasma:
                    epsr1 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                    epsr2 = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_2][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr1 = self.Rho_Parameterization(rho, bounds)
                    epsr2 = self.Rho_Parameterization(rho, bounds)
                sim1.eps_r = epsr1
                sim2.eps_r = epsr2

                _, _, Ez1 = sim1.solve(self.sources[src_1][0])
                _, _, Ez2 = sim2.solve(self.sources[src_2][0])

                return (mode_overlap(Ez1, self.probes[prb_1][0])/E01)*\
                       (mode_overlap(Ez2, self.probes[prb_2][0])/E02)-\
                       (field_mag_int(Ez1, self.probes[prb_2][3])/E01l)*\
                       (field_mag_int(Ez2, self.probes[prb_1][3])/E02l)

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        else:
            raise RuntimeError("The two sources must have the same polarization.")


    def Optimize_Logic_Gate(self, Rho, src_1, src_2, src_c, prb_n, prb_t, alpha,\
            nepochs, logic, bounds = [], plasma = False, wp_max = 0):
        """
        Optimize a logic gate PMM

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            src_c: Key for the constant source in the sources dict.
            prb_n: Key for the not true probe in the probes dict
            prb_t: Key for the true probe in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            logic: string specifying what kind of logic you're interested in
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
        """
        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init = self.Rho_Parameterization(Rho, bounds)
            
            sim = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            
            Exc, _, _ = sim.solve(self.sources[src_c][0])
            Ex1, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
            Ex2, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
            Ex12, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                    self.sources[src_2][0])

            Ec0n = mode_overlap(Exc, self.probes[prb_n][0])
            E10n = mode_overlap(Ex1, self.probes[prb_n][0])
            E20n = mode_overlap(Ex2, self.probes[prb_n][0])
            E10t = mode_overlap(Ex1, self.probes[prb_t][0])
            E20t = mode_overlap(Ex2, self.probes[prb_t][0])
            E120t = mode_overlap(Ex12, self.probes[prb_t][0])
            E10ln = field_mag_int(Ex1, self.probes[prb_n][3])
            E20ln = field_mag_int(Ex2, self.probes[prb_n][3])
            E120ln = field_mag_int(Ex12, self.probes[prb_n][3])
            Ec0lt = field_mag_int(Exc, self.probes[prb_t][3])
            E10lt = field_mag_int(Ex1, self.probes[prb_t][3])
            E20lt = field_mag_int(Ex2, self.probes[prb_t][3])
           
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                Exc, _, _ = sim.solve(self.sources[src_c][0])
                Ex1, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
                Ex2, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
                Ex12, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                       self.sources[src_2][0])

                if logic == 'and':
                    off = mode_overlap(Exc, self.probes[prb_n][0])/Ec0n -\
                        field_mag_int(Exc, self.probes[prb_t][3])/Ec0lt
                    one = mode_overlap(Ex1, self.probes[prb_n][0])/E10n -\
                        field_mag_int(Ex1, self.probes[prb_t][3])/E10lt
                    two = mode_overlap(Ex2, self.probes[prb_n][0])/E20n -\
                        field_mag_int(Ex2, self.probes[prb_t][3])/E20lt
                    both = 3*mode_overlap(Ex12, self.probes[prb_t][0])/E120t -\
                        3*field_mag_int(Ex12, self.probes[prb_n][3])/E120ln
                            
                elif logic == 'or':
                    off = 3*mode_overlap(Exc, self.probes[prb_n][0])/Ec0n -\
                        3*field_mag_int(Exc, self.probes[prb_t][3])/Ec0lt
                    one = mode_overlap(Ex1, self.probes[prb_t][0])/E10t -\
                        field_mag_int(Ex1, self.probes[prb_n][3])/E10ln
                    two = mode_overlap(Ex2, self.probes[prb_t][0])/E20t -\
                        field_mag_int(Ex2, self.probes[prb_n][3])/E20ln
                    both = mode_overlap(Ex12, self.probes[prb_t][0])/E120t -\
                        field_mag_int(Ex12, self.probes[prb_n][3])/E120ln

                else:
                    raise RuntimeError("Logic not implemented yet")

                return off + one + two + both

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj

        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            #Begin by running sim with initial parameters to get normalization consts
            if plasma:
                epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
            else:
                epsr_init = self.Rho_Parameterization(Rho, bounds)
            
            sim = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            
            _, _, Ezc = sim.solve(self.sources[src_c][0])
            _, _, Ez1 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
            _, _, Ez2 = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
            _, _, Ez12 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                    self.sources[src_2][0])

            Ec0n = mode_overlap(Ezc, self.probes[prb_n][0])
            E10n = mode_overlap(Ez1, self.probes[prb_n][0])
            E20n = mode_overlap(Ez2, self.probes[prb_n][0])
            E10t = mode_overlap(Ez1, self.probes[prb_t][0])
            E20t = mode_overlap(Ez2, self.probes[prb_t][0])
            E120t = mode_overlap(Ez12, self.probes[prb_t][0])
            E10ln = field_mag_int(Ez1, self.probes[prb_n][3])
            E20ln = field_mag_int(Ez2, self.probes[prb_n][3])
            E120ln = field_mag_int(Ez12, self.probes[prb_n][3])
            Ec0lt = field_mag_int(Ezc, self.probes[prb_t][3])
            E10lt = field_mag_int(Ez1, self.probes[prb_t][3])
            E20lt = field_mag_int(Ez2, self.probes[prb_t][3])
            
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
                if plasma:
                    epsr = self.Rho_Parameterization_wp(rho,\
                            self.sources[src_1][1]*self.a/2/np.pi/c, wp_max)
                else:
                    epsr = self.Rho_Parameterization(rho, bounds)
                sim.eps_r = epsr

                _, _, Ezc = sim.solve(self.sources[src_c][0])
                _, _, Ez1 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
                _, _, Ez2 = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
                _, _, Ez12 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                       self.sources[src_2][0])

                if logic == 'and':
                    off = 6*mode_overlap(Ezc, self.probes[prb_n][0])/Ec0n -\
                        field_mag_int(Ezc, self.probes[prb_t][3])/Ec0lt
                    one = mode_overlap(Ez1, self.probes[prb_n][0])/E10n -\
                        150*field_mag_int(Ez1, self.probes[prb_t][3])/E10lt
                    two = mode_overlap(Ez2, self.probes[prb_n][0])/E20n -\
                        100*field_mag_int(Ez2, self.probes[prb_t][3])/E20lt
                    both = 7*mode_overlap(Ez12, self.probes[prb_t][0])/E120t -\
                        100*field_mag_int(Ez12, self.probes[prb_n][3])/E120ln
                            
                elif logic == 'or':
                    off = 3*mode_overlap(Ezc, self.probes[prb_n][0])/Ec0n -\
                        3*field_mag_int(Ezc, self.probes[prb_t][3])/Ec0lt
                    one = mode_overlap(Ez1, self.probes[prb_t][0])/E10t -\
                        field_mag_int(Ez1, self.probes[prb_n][3])/E10ln
                    two = mode_overlap(Ez2, self.probes[prb_t][0])/E20t -\
                        field_mag_int(Ez2, self.probes[prb_n][3])/E20ln
                    both = mode_overlap(Ez12, self.probes[prb_t][0])/E120t -\
                        field_mag_int(Ez12, self.probes[prb_n][3])/E120ln

                else:
                    raise RuntimeError("Logic not implemented yet")

                return off + one + two + both

            # Compute the gradient of the objective function
            objective_jac = jacobian(objective, mode='reverse')

            # Maximize the objective function using an ADAM optimizer
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs, bounds=bounds,\
                                direction = 'max', step_size = alpha)

            return rho_optimum.reshape(Rho.shape), obj


    def Params_to_Exp(self, rho, src, bounds = [], plasma = False, nu_col=0):
        """
        Output experimental data needed to rebuild a certain design

        Args:
            rho: parameters for trainable element permittivity values
            src: key for active source in sources dict
            bounds: max and min perm values for training
            plasma: bool specifying if params map to wp
            nu_col: supposed collision frequency in c/a units
        """
        if plasma:
            self.Params_to_Exp_wp(rho, src)
        else:
            print("The lattice frequency is: ", c/self.a/(10**9)," GHz")
            print("The source frequency is: ", self.sources[src][1]/2/np.pi/(10**9), " GHz")
            print("The plasma frequencies (GHz) necessary to achieve this design are:")
            print(np.sqrt((1-np.real(self.Rho_to_Eps(rho, bounds)))*\
                (self.sources[src][1]**2+(nu_col*2*np.pi*c/self.a)**2))/(10**9))


    def Params_to_Exp_wp(self, rho, src, nu_col=0):
        """
        Output experimental data needed to rebuild a certain design

        Args:
            rho: parameters for trainable element permittivity values
            bounds: max and min perm values for training
            src: key for active source in sources dict
            nu_col: supposed collision frequency in c/a units
        """
        print("The lattice frequency is: ", c/self.a/(10**9)," GHz")
        print("The source frequency is: ", self.sources[src][1]/2/np.pi/(10**9), " GHz")
        print("The plasma frequencies (GHz) necessary to achieve this design are:")
        print(rho*c/self.a/(10**9))


    def Save_Params(self, rho, savepath):
        """
        Saves optimization parameters. A wrapper for np.savetxt

        Args:
            rho: parameters to be saved
            savepath: save path. Must be csv.
        """
        np.savetxt(savepath, rho, delimiter=",")


    def Read_Params(self, readpath):
        """
        Reads optimization paramters.

        Args:
            readpath: read path. Must be csv.
        """
        return np.loadtxt(readpath, delimiter=",")
