'''Ion String Tools
Provides functions for:
    * Calculating ion positions
    * Calculating vibrational modes
    * Calculating crosstalk intensities
'''

import numpy as np
from scipy.optimize import fsolve, leastsq
from math import exp, pi as π

# Physical Constants
physconsts = {
    "c" : 299792458, # speed of light, units of m/s
    "ϵ0" : 8.854188e-12, # permittivity of free space, units of F/m
    "amu" : 1.66054e-27, # atomic mass unit, units of kg
    "q_e" : 1.60218e-19 # electron charge, units of C
}
physconsts["mass_Ca40"] = 39.9626*physconsts["amu"] # mass of calcium atom, units of kg

def ion_position_potential(x):
    '''Potential energy of the ion string as a function of the positions of the ions
        
    Params
        x : list
            the positions of the ions, in units of the length scale

    Returns
        float
        potential energy of the string, in units of the energy scale
    '''
    N = len(x)
    return [x[m] - sum([1/(x[m]-x[n])**2 for n in range(m)]) + sum([1/(x[m]-x[n])**2 for n in range(m+1,N)])
               for m in range(N)]

def calcPositions(N):
    '''Calculate the equilibrium ion positions
    
    Params
        N : int
            number of ions
    
    Returns
        list
        equilibrium positions of the ions
    
    '''
    estimated_extreme = 0.481*N**0.765 # Hardcoded, should work for at least up to 50 ions
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    '''Calculate the length scale for the trap
    
    Params
        ν : float
            trap frequency, in units of radians/sec
        M : float
            mass of ion, in units of kg
        Z : int
            degree of ionization (net charge on ion)
        
    Returns
        float
        length scale in units of meters
    '''
    if M==None: M = physconsts["mass_Ca40"]
    if Z==None: Z = 1
    return ((Z**2*physconsts["q_e"]**2)/(4*π*physconsts['ϵ0']*M*ν**2))**(1/3)

def calcAxialModes(N, masses=None):
    '''Calculate axial vibrational modes
    
    Params
        N : int
            number of ions
        masses : list
            masses of ions
    
    Returns
        list
        vibrational modes of string, each encoded in a tuple (frequency, mode vector). Frequency is in units of the COM frequency.
    '''
    u = calcPositions(N)
    if masses is None: masses = [1 for _ in range(N)]
    A = [
            [1 + 2*sum([1/abs(u[m]-u[p])**3 if p!=m else 0 for p in range(0, N)]) if n==m
             else -2/abs(u[m]-u[n])**3
             for m in range(0,N)]
        for n in range(0,N)]
    A = np.array(A)
    for i in range(N):
        for j in range(N):
            A[i,j] /= np.sqrt(masses[i])*np.sqrt(masses[j])
    eigvals, eigvecs = np.linalg.eig(A)
    freqs = np.sqrt(eigvals)
    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0])
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/np.sqrt(masses[i]) for i in range(N)])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes


def calcRadialModes(N, masses=None, νratio=3):
    '''Calculate transverse vibrational modes
    
    Params
        N : int
            number of ions
    
    Returns
        list
        vibrational modes of string, each encoded in a tuple (frequency, mode vector). Frequency is in units of the COM frequency.
    '''
    if masses is None: masses = [1 for _ in range(N)]
    ueq = calcPositions(N)
    A = np.zeros((N, N))
    for i in range(N):
        A[i][i] = -νratio**2 + sum([1/(ueq[i]-ueq[m])**3 for m in range(0, i)]) + sum([1/(ueq[m]-ueq[i])**3 for m in range(i+1, N)])
        for j in range(0, i):
            A[i][j] = -1/(ueq[i]-ueq[j])**3
        for j in range(i+1, N):
            A[i][j] = -1/(ueq[j]-ueq[i])**3
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals *= -1
    invalid_modes = np.where(eigvals<0)
    eigvals = np.delete(eigvals, invalid_modes)
    eigvecs = np.delete(eigvecs, invalid_modes, axis=0)
    freqs = np.sqrt(eigvals)
    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0])
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/np.sqrt(masses[i]) for i in range(len(eigvals))])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes

def calcCrosstalkIntensities(N, targets, bw):
    '''Calculate crosstalk intensities at ion positions
    
    Params
        N : int
            number of ions
        targets : list
            indices of target ions, which define the centers of the laser beams
        bw : float
            laser beamwidth
    
    Returns
        list
        total laser intensity at ion positions, intensity of 1 corresponding to intensity at center of single beam
    '''
    positions = calcPositions(N)
    beamcenters = [positions[t] for t in targets]
    intensities = calcCrosstalkIntensitiesAtPositions(beamcenters, bw, positions)
    return intensities
    
    
def calcCrosstalkIntensitiesAtPositions(beamcenters, bw, positions):
    '''Calculate crosstalk intensities at given positions
    
    Params
        beamcenters : list
            positions of the centers of each laser beam
        bw : float
            laser beamwidth (w0 beam radius)
        positions : list
            positions at which to evaluate total laser intensity
            
    Returns
        list
        total laser intensity at given positions, intensity of 1 corresponding to intensity at center of single beam
    
    '''
    intensities = sum([np.array([exp(-((p-c)/bw)**2) for p in positions]) for c in beamcenters])**2
    #intensities = sum([np.array([exp(-2*((p-c)/bw)**2) for p in positions]) for c in beamcenters])
    return intensities