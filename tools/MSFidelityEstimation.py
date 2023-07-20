'''MS Fidelity Estimation
Provides functions for estimating fidelity of MS gate under various errors/approximations.
'''

import numpy as np
from numpy import cos, sin, exp
from itertools import combinations
from operator import mul, add
from functools import reduce
from qutip import sigmax, tensor, basis, fidelity, qeye

import tools.IonChainTools as ict

π = np.pi
pi = np.pi

def estimateMSFidelity_errgate_smallTN(N, m, targets, bw=1, mode_type="axial", mvec=None, intensities=None, νratio=None):
    '''
    Estimate fidelity of MS operation using error gate model considering only small error gates between
    target and neighbor ions.
    
    Params
        N : number of ions in string
        m : mode number (COM is 0)
        i : index of first ion (index starts at 0)
        j : index of second ion (index starts at 0)
        bw_ratio : beamwidth as multiple of minumum ion spacing
    '''
    i, j = targets
    if N%2==1 and m%2==1 and (i==(N-1)/2 or j==(N-1)/2):    # Ignore impossible operations (zero coupling)
        return -1
    neighbors = [n for n in range(N) if n in (i-1, i+1, j-1, j+1) and n not in (i, j)]
    if mvec is None:
        if mode_type=="axial":
            mvec = ict.calcAxialModes(N)[m][1]
        elif mode_type=="radial":
            mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
    if intensities is None: intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
    Ω = lambda i1,i2 : np.sqrt(intensities[i1]*intensities[i2])*mvec[i1]*mvec[i2]
    error = 0
    for target in i,j:
        for neigh in neighbors:
            error += np.pi**2/16*(Ω(target,neigh)/Ω(i,j))**2
    return 1-error


def estimateMSFidelity_errgate_exactTN(N, m, targets, bw=1, mode_type="axial", mvec=None, intensities=None, νratio=None):
    '''
    Estimate fidelity of MS operation using error gate model considering error gates of any size between
    target and neighbor ions.
    
    Params
        N : number of ions in string
        m : mode number (COM is 0)
        i : index of first ion (index starts at 0)
        j : index of second ion (index starts at 0)
        bw_ratio : beamwidth as multiple of minumum ion spacing
    '''
    i, j = targets
    if N%2==1 and m%2==1 and (i==(N-1)/2 or j==(N-1)/2):    # Ignore impossible operations (zero coupling)
        return -1
    neighbors = [n for n in range(N) if n in (i-1, i+1, j-1, j+1) and n not in (i, j)]
    if mvec is None:
        if mode_type=="axial":
            mvec = ict.calcAxialModes(N)[m][1]
        elif mode_type=="radial":
            mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
    if intensities is None: intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
    Ω = lambda i1,i2 : np.sqrt(intensities[i1]*intensities[i2])*mvec[i1]*mvec[i2]
    tr = 1
    gatefreq = Ω(i, j)
    # Doubles
    for target in i,j:
        for neigh in neighbors:
            #print(Ω(target, neigh)/gatefreq)
            tr *= cos(pi/4*Ω(target, neigh)/gatefreq)
    #print("Just TN Doubles:", tr**2)
    # Loops
    # Iterate of pairs of neighbors
    for n1i in range(len(neighbors)):
        for n2i in range(n1i+1, len(neighbors)):
            n1 = neighbors[n1i]
            n2 = neighbors[n2i]
            looptr = 1
            for target in i,j:
                for neigh in neighbors:
                    if neigh in (n1, n2):
                        looptr *= sin(pi/4*Ω(target, neigh)/gatefreq)
                    else:
                        looptr *= cos(pi/4*Ω(target, neigh)/gatefreq)
            tr += looptr
    if len(neighbors) ==4: # No full loop for 1 or 3 neighbors
        alllooptr = 1
        for target in i,j:
            for neigh in neighbors:
                alllooptr *= sin(pi/4*Ω(target, neigh)/gatefreq)
        tr += alllooptr
    fid = tr**2
    return fid


def estimateMSFidelity_errgate_allnearpairs(N, m, targets, bw, mvec=None, mode_type="axial", intensities=None, ψ0=None, moreneigh=False, νratio=None):
    '''
    Estimates error in 2-ion MS gate using error gate model. Simulates all error gates (including between pairs of
    neighbors), for either nearest neighbors or for next-nearest neighbors as well.
    '''
    i, j = targets # Ions to perform gate between
    neighbors = [a for a in range(N) if a in (i-1, i+1, j-1, j+1) and a not in targets] # Nearest neighbors
    # Nearest neighbors and next nearest neighbors
    moreneighbors = [a for a in range(N) if a in (i-1, i-2, i+1, i+2, j-1, j-2, j+1, j+2) and a not in targets]
    # If requested use both nearest and next nearest neighbors
    if moreneigh:
        neighbors = moreneighbors
    # All neighbors to be considered, plus the targets themselves
    allions = neighbors + list(targets)
    # Dictionary to go from index of ion in full chain to index of ion in truncated space that is simulated
    simindex = dict([reversed(pair) for pair in enumerate(allions)])
    # Dictionary to go from index in truncated simulated space to index in full chain
    realindex = dict(enumerate(allions))
    simN = len(allions)    # Only simulate targets and neighbors, not unaffected ions
    if ψ0==None: # If unset, set initial state to ground state
        ψ0 = tensor([basis(2,0) for _ in range(simN)])
    if mvec is None: # If no mode vector given, calculate it according to given arguments
        if mode_type=="axial":
            mvec = ict.calcAxialModes(N)[m][1]
        elif mode_type=="radial":
            mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
    if intensities is None: # If laser intensities not given, calculate it according to given arguements
        intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
    # Speed of gate between target ions is proportional to electric fields and move coupling values for target ions
    target_gatespeed_unscaled = np.sqrt(intensities[i])*mvec[i]*np.sqrt(intensities[j])*mvec[j]
    # ϵ described effective angle rotated by error gate during gate time
    ϵ = lambda a, b : π/4*np.sqrt(intensities[a])*mvec[a]*np.sqrt(intensities[b])*mvec[b]/target_gatespeed_unscaled
    # X_i is a pauli-X gate acting only on ion i. It acts on the truncated hilbert space being simulated
    X = lambda i: tensor([sigmax() if simj==simindex[i] else qeye(2) for simj in range(simN)])
    # Target gate is XX gate of π/4, ϵ(i,j) should be π/4
    target_gate = (1j*ϵ(i,j)*X(i)*X(j)).expm()
    # Apply target gate to initial state
    target_state = target_gate*ψ0
    # Error gate ion pairs are all pairs of ions to be simulated other than the target ions
    errorgate_ionpairs = [(allions[i1],allions[i2])
                          for i1 in range(len(allions)) for i2 in range(i1+1, len(allions))
                          if set((allions[i1],allions[i2]))!=set(targets)]
    # If there are no error gates, (for example, when there are only 2 ions), then fidelity is 1
    if len(errorgate_ionpairs) == 0:
        return 1
    # Add generators of error gates (becomes a product when exponentiated)
    errgate_sum = sum([ϵ(a,b)*X(a)*X(b) for a,b in errorgate_ionpairs])
    # Exponentiate to make unitary operation representing all error
    errors_gate = (1j*sum([ϵ(a,b)*X(a)*X(b) for a,b in errorgate_ionpairs])).expm()
    # Apply error gate to target state and find inner product norm with target state
    fid = (target_state.dag()*errors_gate*target_state).norm()**2
    return fid