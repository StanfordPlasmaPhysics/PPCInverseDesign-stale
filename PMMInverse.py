"""
The module composed in this file is meant to serve as a platform for
designing plasma metamaterial devices and then optimizing the plasma density
of the elements composing the metamaterial to achieve a certain functionality.
It is built atop ceviche, an autograd compliant FDFD EM simulation tool 
(https://github.com/fancompute/ceviche).
Jesse A Rodriguez, 09/15/2020
"""

import numpy as np
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
from autograd.scipy.signal import convolve as conv
from skimage.draw import disk, rectangle
import ceviche
from ceviche import fdfd_ez, jacobian, fdfd_hz
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
import collections
from make_gif import make_gif

## Ahmed's plotting tools #####################################################
# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side 
    from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
def real(val, outline=None, ax=None, cbar=False, cmap='RdBu', outline_alpha=0.5,\
         vmin=None, vmax=None):
    """
    Plots the real part of 'val', optionally overlaying an outline of 'outline'
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    
    if vmin is None:
        vmax = np.real(val).max()
        vmin = np.real(val).min()
    h = ax.imshow(np.real(val.T), cmap=cmap, origin='lower', clim=(vmin, vmax),\
                  norm=MidpointNormalize(midpoint=0,vmin=vmin,vmax=vmax))
    
    if outline is not None:
        ax.contour(outline.T, 0, colors='k', alpha=outline_alpha)

    if cbar:
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_ticks([-6, 0, 1])
        cbar.set_ticklabels(['-6', '0', '1'])
        cbar.ax.set_ylabel('Relative Permittivity')
    
    return ax

def abslt(val, outline=None, ax=None, cbar=False, cmap='magma', outline_alpha=0.5,\
          outline_val=None, vmax=None):
    """
    Plots the absolute value of 'val', optionally overlaying an outline of 
    'outline'
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)      
    
    if vmax is None:
        vmax = np.abs(val).max()
    h = ax.imshow(np.abs(val.T), cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    
    if outline_val is None and outline is not None: 
        outline_val = 0.5*(outline.min()+outline.max())
    if outline is not None:
        ax.contour(outline.T, [outline_val], colors='w', alpha=outline_alpha)

    if cbar:
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_ticks([])
        cbar.ax.set_ylabel('Electric Field Strength')
    
    return ax
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
        self.sources = {} #Empty dict to hold source arrays
        self.probes = {} #Empty dict to hold probe arrays

    def Add_Rod(self, r, x, y, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            r: radius of the rod in a units
            x: x-coordinate in a units
            y: y-coordinate in a units
            eps: relative permittivity of the rod
        """
        R = int(round(r*self.res))
        X = int(round(x*self.res))
        Y = int(round(y*self.res))
        rr, cc = disk((X, Y), R, shape = self.epsr.shape)
        
        self.epsr[rr, cc] = eps


    def Add_Block(self, w, h, x, y, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            w: width (x-dir) of the block in a units
            h: height (y-dir) of the block in a units
            x: x-coordinate of the bottom left corner in a units
            y: y-coordinate of the bottom left corner in a units
            eps: relative permittivity of the block
        """
        H = int(round(h*self.res))
        W = int(round(w*self.res))
        X = int(round(x*self.res))
        Y = int(round(y*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.epsr[rr, cc] = eps


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
        

    def Add_Probe(self, prb_x, prb_y, prb_name):
        """
        Add a probe to the domain.

        Args:
            prb_x: 1-d array containing the x-coords of of the probe pixels
            prb_y: 1-d array containing the y-coords of of the probe pixels
            src_name: string that serves as key for the probe in the dict
        """
        probe = np.zeros((self.Nx, self.Ny), dtype=np.complex)
        source[src_x, src_y] = 1

        self.sources[src_name] = source


    def Rod_Array(self, r, x_start, y_start, nrods_x, nrods_y, rod_eps,\
                  d_x = 1, d_y = 1):
        """
        Add a 2D rectangular rod array to the domain. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            nrods_x: number of rods in x-direction
            nrods_y: number of rods in y-direction
            rod_eps: np array of size nrods_x by nrods_y giving the relative 
                     permittivity of each of the rods
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
        """
        for i in range(nrods_x):
            for j in range(nrods_y):
                x = x_start + i*d_x
                y = y_start + j*d_y
                self.Add_Rod(r, x, y, rod_eps[i,j])


    def Viz_Sim_abs(self, src_name, slices=[]):
        """
        Solve and visualize a simulation with a certain source active
        
        Args:
            src_name: string that indicates which source you'd like to simulate
        """
        if self.sources[src_name][2] == 'hz':
            simulation = fdfd_hz(self.sources[src_name][1], self.dl, self.epsr,\
                         [self.Npml, self.Npml])
            Ex, Ey, Hz = simulation.solve(self.sources[src_name][0])
            
            fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
            ceviche.viz.abs(Hz, outline=self.epsr, ax=ax[0], cbar=False)
            for sl in slices:
                ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
            ceviche.viz.abs(self.epsr, ax=ax[1], cmap='Greys');
            plt.show()

            return (simulation, ax)

        elif self.sources[src_name][2] == 'ez':
            simulation = fdfd_ez(self.sources[src_name][1], self.dl, self.epsr,\
                         [self.Npml, self.Npml])
            Hx, Hy, Ez = simulation.solve(self.sources[src_name][0])

            fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
            ceviche.viz.abs(Ez, outline=self.epsr, ax=ax[0], cbar=False)
            for sl in slices:
                ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
            ceviche.viz.abs(self.epsr, ax=ax[1], cmap='Greys');
            plt.show()

            return (simulation, ax)

        else:
            raise RuntimeError('The polarization associated with this source is\
                                not valid.')
