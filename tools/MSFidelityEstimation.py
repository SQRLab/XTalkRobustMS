'''MS Fidelity Estimation
Provides functions for estimating fidelity of MS gate under various
errors/approximations.
'''

import numpy as np
from numpy import cos, sin, exp, sqrt
from math import factorial
from itertools import combinations
import itertools
from operator import mul, add
from functools import reduce
from qutip import sigmax, tensor, basis, fidelity, qeye
import qutip as qtp
import scipy.constants
import sys, os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import IonChainTools as ict

π = np.pi
pi = np.pi

###############################################################################
# Generic Calculations
###############################################################################

def Ωvals_from_fixed_neighbor_xtalk(N, targets, neighΩ, nneighΩ=0):
    Ωvals = np.zeros(N)
    for t in targets:
        Ωvals[t] +=1
        for n in (t-1, t+1):
            if n>=0 and n<N:
                Ωvals[n] += neighΩ
        for nn in (t-2, t+2):
            if nn>=0 and nn<N:
                Ωvals[nn] += nneighΩ
    return Ωvals

def coherent_coeff(α, n):
    return exp(-abs(α)**2/2)*α**n/sqrt(factorial(n))

def σ(ϕ):
    return np.cos(ϕ)*qtp.sigmax()+np.sin(ϕ)*qtp.sigmay()

###############################################################################
# Parameter Organization Classes
###############################################################################

class VibMode:
    def __init__(self, N, freq, mvec):
        self.N = N
        self.freq = freq
        self.mvec = mvec

    def __repr__(self):
        return str({'N':self.N, 'freq':self.freq, 'mvec':self.mvec})

    @classmethod
    def fromDict(cls, d):
        return cls(d['N'], d['freq'], d['mvec'])
        

class TrapSpec:
    def __init__(self, νz, νr):
        self.νz = νz
        self.νr = νr
        
    def __repr__(self):
        return str({'νz':self.νz, 'νr':self.νr})

    @classmethod
    def fromDict(cls, d):
        return cls(d['νz'], d['νr'])
        
        
class MSLaserSpec:
    def __init__(self, ωd, Ωvals, ϕB, ϕR):
        self.ωd = ωd
        self.Ωvals = Ωvals
        self.ϕB = ϕB
        self.ϕR = ϕR
        
    def __repr__(self):
        # Convert Ωvals to list in case it's a numpy array so that it converts to string like a primitive
        return str({'ωd':self.ωd, 'Ωvals':list(self.Ωvals), 'ϕB':self.ϕB, 'ϕR':self.ϕR})

    @classmethod
    def fromDict(cls, d):
        return cls(d['ωd'], d['Ωvals'], d['ϕB'], d['ϕR'])
        
        
class TIDeviceSpec:
    def __init__(self, N, trapspec, ωqbit, ωraman=0, M=1):
        self.N = N
        self.trapspec = trapspec
        self.ωqbit = ωqbit
        self.ωraman = ωraman
        self.M = M # in kg
        self._modes = {'axial':None, 'radial':None}
        
    def modes(self, modetype):
        if self._modes[modetype] != None: return self._modes[modetype]
        if modetype=='radial':
            modedata = ict.calcRadialModes(self.N, νratio=self.trapspec.νr/self.trapspec.νz)
        elif modetype=='axial':
            modedata = ict.calcAxialModes(self.N)
        else:
            raise Exception("Only modetype options are 'radial' and 'axial'")
        modes = [VibMode(self.N, modedata[m][0]*self.trapspec.νz, modedata[m][1]) for m in range(self.N)]
        self._modes[modetype] = modes
        return modes
    
    def LDparam(self, m, modetype, i):
        νm = self.modes(modetype)[m].freq
        z0 = np.sqrt(scipy.constants.hbar/(2*self.M*νm))
        if self.ωraman != 0:
            k = (2*self.ωraman+self.ωqbit)/scipy.constants.c
        else:
            k = self.ωqbit/scipy.constants.c
        bmi = self.modes(modetype)[m].mvec[i] # Coupling of ion i to mode m, usually denoted $b^m_i$
        return k*z0*bmi
    
    def __repr__(self):
        return str({'N':self.N, 'trapspec':self.trapspec, 'ωqbit':self.ωqbit, 'ωraman':self.ωraman, 'M':self.M})

    @classmethod
    def fromDict(cls, d):
        return cls(d['N'], TrapSpec.fromDict(d['trapspec']), d['ωqbit'], d['ωraman'], d['M'])

class MSOpSpec:
    def __init__(self, devicespec, mslaserspec, duration, targets=None, m=None, modetype=None, K=None):
        self.devicespec = devicespec
        self.mslaserspec = mslaserspec
        self.duration = duration
        self.targets = targets
        self.m = m
        self.modetype = modetype
        self.K = K
        
    def __repr__(self):
        return str({'devicespec':self.devicespec, 'mslaserspec':self.mslaserspec, 'duration':self.duration,\
                    'targets':self.targets, 'm':self.m, 'modetype':self.modetype, 'K':self.K})

    @classmethod
    def fromDict(cls, d):
        return cls(TIDeviceSpec.fromDict(d['devicespec']), MSLaserSpec.fromDict(d['mslaserspec']),\
                   d['duration'], d['targets'], d['m'], d['modetype'], d['K'])


class MSFidCalculation:
    def __init__(self, opspec=None, method=None, fid=None, fidions=None):
        self.opspec = opspec
        self.method = method
        self.fid = fid
        self.fidions = fidions

    def __repr__(self):
        return str({'opspec':self.opspec, 'method':self.method, 'fid':self.fid, 'fidions':self.fidions})

    @classmethod
    def fromDict(cls, d):
        return cls(MSOpSpec.fromDict(d['opspec']), d['method'], d['fid'], d['fidions'])
        

###############################################################################
# Generic MS Calculations
###############################################################################

def calc_necessary_sideband_detuning_and_rabi_freq(devicespec, targets, m, modetype, τ, K=1):
    δ = 2*π*K/τ # Detuning is restricted by requirement to close phase-space loop
    νm = devicespec.modes(modetype)[m].freq
    ηmt0 = devicespec.LDparam(m, modetype, targets[0])
    ηmt1 = devicespec.LDparam(m, modetype, targets[1])
    Ω = δ/np.sqrt(abs(4*K*ηmt0*ηmt1)) # Single rabi freq, meant for both target ions
    return νm+δ, Ω

def calc_necessary_sideband_detuning_and_gate_time(devicespec, targets, Ω, m, modetype, K=1):
    mode = devicespec.getModes(modetype)[m]
    ηmt0 = devicespec.LDparam(m, modetype, targets[0])
    ηmt1 = devicespec.LDparam(m, modetype, targets[1])
    δ = sqrt(abs(4*K*Ω[t1]*Ω[t2]*ηmt1*ηmt2))
    τ = 2*π*K/δ
    ωd = mode.freq + δ
    return ωd, τ


def parity(ρ):
    return ρ[0,0]+ρ[3,3]-ρ[1,1]-ρ[2,2]


def parityCurve(ρ, numpoints=100):
    R = lambda ϕ : 1/sqrt(2)*qtp.Qobj([[1, -1j*exp(-1j*ϕ)], [-1j*exp(1j*ϕ), 1]])
    parities = []
    for ϕ in np.linspace(0,π,numpoints):
        RtensorR = qtp.tensor(R(ϕ),R(ϕ))
        parities.append((ϕ,parity(RtensorR*ρ*RtensorR.dag())))
    return np.array(parities)

###############################################################################
# MS Operation Calculator
###############################################################################


def MS_Analytical_zeroinitalstate_tracemodes(opspec, simmodes=None, modetype='radial', modetrunc=2):
    devicespec = opspec.devicespec
    N = devicespec.N
    τ = opspec.duration
    Ωvals = opspec.mslaserspec.Ωvals
    ωd = opspec.mslaserspec.ωd
    if opspec.modetype==None: modetype = modetype
    else: modetype = opspec.modetype
    ϕB = opspec.mslaserspec.ϕB
    ϕR = opspec.mslaserspec.ϕR
    ϕs = (ϕB+ϕR)/2
    ϕm = (ϕB-ϕR)/2
    
    if simmodes == None:
        simmodes = list(range(N))

    illuminated = [i for i in range(N) if Ωvals[i] != 0] # All illuminated ions
    Ωvals_nonzero = [Ωvals[i] for i in illuminated]
    Neff = len(illuminated) # We only need to work on subspace of illuminated qubits, dimension of effective N

    # Effect of M1 term
    α_kλ = np.zeros((len(simmodes),2**Neff), dtype=np.cdouble) # Mode displacements in each mode k for each qubit state λ, comes from M1 term
    for keff, k in enumerate(simmodes):
        δk = ωd - devicespec.modes(modetype)[k].freq
        α_i = np.zeros(Neff, np.cdouble)
        for ieff, i in enumerate(illuminated):
            α_i[ieff] = Ωvals[i]*devicespec.LDparam(k,modetype,i)/(2*δk)*(exp(-1j*δk*τ)-1) # Leave out exp(iϕm) since we trace out modes anyway
        for l, λ in enumerate(itertools.product(*([[-1,+1]]*Neff))):
            α_kλ[keff,l] += np.sum(α_i*λ)

    # Effect of M2 term
    θλ = np.zeros(2**Neff) # Angle of phase placed on each qubit state λ, comes from M2 term
    θ_j1j2 = np.zeros((Neff, Neff))
    for keff, k in enumerate(simmodes):
        δk = ωd - devicespec.modes(modetype)[k].freq
        for (j1eff, j1), (j2eff, j2) in itertools.combinations(enumerate(illuminated),2):
            modekcontrib = -1/4*Ωvals[j1]*Ωvals[j2]*devicespec.LDparam(k,modetype,j1)*devicespec.LDparam(k,modetype,j2)
            modekcontrib *= (τ/δk - sin(δk*τ)/δk**2)
            θ_j1j2[j1eff,j2eff] += modekcontrib
    for l, λ in enumerate(itertools.product(*([[-1,+1]]*Neff))):
        for j1eff, j2eff in itertools.combinations(range(Neff),2):
            θλ[l] += λ[j1eff]*λ[j2eff]*θ_j1j2[j1eff,j2eff] * (2 if j1eff!=j2eff else 1)
    
    # Creating qubit density matrix (only illuminated qubits)
    ρqbit_mat = np.ones((2**Neff,2**Neff), dtype=np.cdouble)/2**Neff # Rotation basis
    for l1, l2 in itertools.product(range(2**Neff), range(2**Neff)):
        ρqbit_mat[l1, l2] *= exp(1j*(θλ[l1]-θλ[l2])) # Rotation phase
        mode_trace_factor = 1
        for keff, k in enumerate(simmodes):
            αl1 = α_kλ[keff,l1]
            αl2 = α_kλ[keff,l2]
            mode_trace_factor *= exp(-1/2*(abs(αl1)**2+abs(αl2)**2-2*αl2.conjugate()*αl1))
        ρqbit_mat[l1, l2] *= mode_trace_factor
    
    ρqbit = qtp.Qobj(ρqbit_mat, dims=[[2]*Neff,[2]*Neff])
    
    σeigstates = σ(π/2+ϕs).eigenstates()
    σbasis = qtp.Qobj(np.c_[σeigstates[1][0].data.A, σeigstates[1][1].data.A],dims=[[2],[2]])
    change_of_basis = qtp.tensor([σbasis]*Neff)
    ρqbit_zbasis = change_of_basis*ρqbit*change_of_basis.dag()
    
    return ρqbit_zbasis, illuminated
    

# ###############################################################################
# # MS Fidelity Calculation Methods **OLD VERSIONS**
# ###############################################################################

# def estimateMSFidelity_errgate_smallTN(N, m, targets, bw=1, mode_type="axial",
#                                     mvec=None, intensities=None, νratio=None):
#     '''
#     Estimate fidelity of MS operation using error gate model considering only
#     small error gates between target and neighbor ions.
    
#     Params
#         N : number of ions in string
#         m : mode number (COM is 0)
#         i : index of first ion (index starts at 0)
#         j : index of second ion (index starts at 0)
#         bw_ratio : beamwidth as multiple of minumum ion spacing
#     '''
#     i, j = targets
#     if N%2==1 and m%2==1 and (i==(N-1)/2 or j==(N-1)/2):    # Ignore impossible operations (zero coupling)
#         return -1
#     neighbors = [n for n in range(N) if n in (i-1, i+1, j-1, j+1) and n not in (i, j)]
#     if mvec is None:
#         if mode_type=="axial":
#             mvec = ict.calcAxialModes(N)[m][1]
#         elif mode_type=="radial":
#             mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
#     if intensities is None: intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
#     Ω = lambda i1,i2 : np.sqrt(intensities[i1]*intensities[i2])*mvec[i1]*mvec[i2]
#     error = 0
#     for target in i,j:
#         for neigh in neighbors:
#             error += np.pi**2/16*(Ω(target,neigh)/Ω(i,j))**2
#     return 1-error


# def estimateMSFidelity_errgate_exactTN(N, m, targets, bw=1, mode_type="axial", mvec=None, intensities=None, νratio=None):
#     '''
#     Estimate fidelity of MS operation using error gate model considering error gates of any size between
#     target and neighbor ions.
    
#     Params
#         N : number of ions in string
#         m : mode number (COM is 0)
#         i : index of first ion (index starts at 0)
#         j : index of second ion (index starts at 0)
#         bw_ratio : beamwidth as multiple of minumum ion spacing
#     '''
#     i, j = targets
#     if N%2==1 and m%2==1 and (i==(N-1)/2 or j==(N-1)/2):    # Ignore impossible operations (zero coupling)
#         return -1
#     neighbors = [n for n in range(N) if n in (i-1, i+1, j-1, j+1) and n not in (i, j)]
#     if mvec is None:
#         if mode_type=="axial":
#             mvec = ict.calcAxialModes(N)[m][1]
#         elif mode_type=="radial":
#             mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
#     if intensities is None: intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
#     Ω = lambda i1,i2 : np.sqrt(intensities[i1]*intensities[i2])*mvec[i1]*mvec[i2]
#     tr = 1
#     gatefreq = Ω(i, j)
#     # Doubles
#     for target in i,j:
#         for neigh in neighbors:
#             #print(Ω(target, neigh)/gatefreq)
#             tr *= cos(pi/4*Ω(target, neigh)/gatefreq)
#     #print("Just TN Doubles:", tr**2)
#     # Loops
#     # Iterate of pairs of neighbors
#     for n1i in range(len(neighbors)):
#         for n2i in range(n1i+1, len(neighbors)):
#             n1 = neighbors[n1i]
#             n2 = neighbors[n2i]
#             looptr = 1
#             for target in i,j:
#                 for neigh in neighbors:
#                     if neigh in (n1, n2):
#                         looptr *= sin(pi/4*Ω(target, neigh)/gatefreq)
#                     else:
#                         looptr *= cos(pi/4*Ω(target, neigh)/gatefreq)
#             tr += looptr
#     if len(neighbors) ==4: # No full loop for 1 or 3 neighbors
#         alllooptr = 1
#         for target in i,j:
#             for neigh in neighbors:
#                 alllooptr *= sin(pi/4*Ω(target, neigh)/gatefreq)
#         tr += alllooptr
#     fid = tr**2
#     return fid


# def estimateMSFidelity_errgate_allnearpairs(N, m, targets, bw=1, mvec=None, mode_type="axial", intensities=None, ψ0=None, moreneigh=False, νratio=None):
#     '''
#     Estimates error in 2-ion MS gate using error gate model. Simulates all error gates (including between pairs of
#     neighbors), for either nearest neighbors or for next-nearest neighbors as well.
#     '''
#     i, j = targets # Ions to perform gate between
#     neighbors = [a for a in range(N) if a in (i-1, i+1, j-1, j+1) and a not in targets] # Nearest neighbors
#     # Nearest neighbors and next nearest neighbors
#     moreneighbors = [a for a in range(N) if a in (i-1, i-2, i+1, i+2, j-1, j-2, j+1, j+2) and a not in targets]
#     # If requested use both nearest and next nearest neighbors
#     if moreneigh:
#         neighbors = moreneighbors
#     # All neighbors to be considered, plus the targets themselves
#     allions = neighbors + list(targets)
#     # Dictionary to go from index of ion in full chain to index of ion in truncated space that is simulated
#     simindex = dict([reversed(pair) for pair in enumerate(allions)])
#     # Dictionary to go from index in truncated simulated space to index in full chain
#     realindex = dict(enumerate(allions))
#     simN = len(allions)    # Only simulate targets and neighbors, not unaffected ions
#     if ψ0==None: # If unset, set initial state to ground state
#         ψ0 = tensor([basis(2,0) for _ in range(simN)])
#     if mvec is None: # If no mode vector given, calculate it according to given arguments
#         if mode_type=="axial":
#             mvec = ict.calcAxialModes(N)[m][1]
#         elif mode_type=="radial":
#             mvec = ict.calcRadialModes(N, νratio=νratio)[m][1]
#     if intensities is None: # If laser intensities not given, calculate it according to given arguements
#         intensities = ict.calcCrosstalkIntensities(N, (i,j), bw)
#     # Speed of gate between target ions is proportional to electric fields and move coupling values for target ions
#     target_gatespeed_unscaled = np.sqrt(intensities[i])*mvec[i]*np.sqrt(intensities[j])*mvec[j]
#     # ϵ described effective angle rotated by error gate during gate time
#     ϵ = lambda a, b : π/4*np.sqrt(intensities[a])*mvec[a]*np.sqrt(intensities[b])*mvec[b]/target_gatespeed_unscaled
#     # X_i is a pauli-X gate acting only on ion i. It acts on the truncated hilbert space being simulated
#     X = lambda i: tensor([sigmax() if simj==simindex[i] else qeye(2) for simj in range(simN)])
#     # Target gate is XX gate of π/4, ϵ(i,j) should be π/4
#     target_gate = (1j*ϵ(i,j)*X(i)*X(j)).expm()
#     # Apply target gate to initial state
#     target_state = target_gate*ψ0
#     # Error gate ion pairs are all pairs of ions to be simulated other than the target ions
#     errorgate_ionpairs = [(allions[i1],allions[i2])
#                           for i1 in range(len(allions)) for i2 in range(i1+1, len(allions))
#                           if set((allions[i1],allions[i2]))!=set(targets)]
#     # If there are no error gates, (for example, when there are only 2 ions), then fidelity is 1
#     if len(errorgate_ionpairs) == 0:
#         return 1
#     # Add generators of error gates (becomes a product when exponentiated)
#     errgate_sum = sum([ϵ(a,b)*X(a)*X(b) for a,b in errorgate_ionpairs])
#     # Exponentiate to make unitary operation representing all error
#     errors_gate = (1j*sum([ϵ(a,b)*X(a)*X(b) for a,b in errorgate_ionpairs])).expm()
#     # Apply error gate to target state and find inner product norm with target state
#     fid = (target_state.dag()*errors_gate*target_state).norm()**2
#     return fid


# def MSUnitary(t, N, Ωvals, η, ωd, modenums=None, modetype="radial", νz=1e6, νr=3e6, ϕB=π/2, ϕB=π/2, modetrunc=5):
#     '''
#     Get unitary matrix describing MS operation
#     '''
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")
    
#     # (Half) Sum and difference of phases of blue and red beams
#     ϕm = (ϕB-ϕR)/2
#     ϕs = (ϕB+ϕR)/2
    
#     # If no modes are specifically included, include all of them
#     # (This can be too much for most computers to handle though, so make sure to specify which modes to use.)
#     if modenums == None:
#         modenums = list(range(N))
                
#     def M1_summation_term(k, i):
#         # M1 models excursion of motional modes and entanglement of motion with qubit states
#         ηik = η*modes[k][1][i] # Lamb-Dicke parameter for ion i
#         νk = modes[k][0]*νz # Frequency of mode k
#         strength = ηik*Ωvals[i]/(2*(ωd-νk)) # "Strength" proportional to displacement of mode, not freq of phase space loop
#         σi = embedop(σ(π/2+ϕs), N, i) # Rotation on ion i, embedded in hilbert space of system
#         # α is time dependent part of phase space displacement of mode k.
#         # NOTE: actual displacement in phase space will have extra factor of strength
#         α = (exp(-1j*(ωd-νk)*t)-1)*exp(1j*ϕm)
#         ak = embedop(destroy(modetrunc), len(modenums), modenums.index(k), modetrunc) # Lowering op on mode k
#         #print('α', α)
#         return tensor(strength*σi, (α*ak - α.conjugate()*ak.dag()))
    
#     def M2_summation_term(k, i1, i2):
#         # M2 is responsible for XX rotations between pairs of ions
#         ηki1 = η*modes[k][1][i1] # Lamb-Dicke parameters for the ions i1 and i2 in the mode k
#         ηki2 = η*modes[k][1][i2]
#         νk = modes[k][0]*νz # Frequency of mode k
#         strength = Ωvals[i1]*Ωvals[i2]*ηki1*ηki2 # Rotation speed is proportional to intensities and LD params for both ions
#         σi1 = embedop(σ(π/2+ϕs), N, i1) # Operators for rotations on i1 and i2, embedded in hilbert space of entire system
#         σi2 = embedop(σ(π/2+ϕs), N, i2)
#         time_dependence = t/(ωd-νk) - sin((ωd-νk)*t)/(ωd-νk)**2 # Dependence on time and detuning from sideband
#         modeeye = tensor([qeye(modetrunc) for _ in modenums]) # Operator to do nothing to modes (M2 does not effect motion)
#         return -1j/4 * strength * tensor(σi1*σi2, modeeye) * time_dependence

#     completeM1 = 0
#     for k in modenums:
#         for i in range(N):
#             completeM1 += M1_summation_term(k, i)
            
#     completeM2 = 0
#     for k in modenums:
#         # Terms with i1=i2 add phase
#         for i in range(N):
#             completeM2 += M2_summation_term(k, i, i)
#         # Terms with i1≠i2 are the same if i1 and i2 are switched, so multiply each by 2 instead of recalculating
#         for i1 in range(N):
#             for i2 in range(i1+1, N):
#                 completeM2 += 2*M2_summation_term(k, i1, i2)
            
#     U = (completeM1+completeM2).expm()

#     return U


# def MSStateFromUnitary(N, m, targets, Ωvals, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L=1):
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")

#     ηmi1 = η*modes[m][1][i1]
#     ηmi2 = η*modes[m][1][i2]

#     # δ = sqrt(abs(2*K*Ωvals[i1]*Ωvals[i2]*ηmi1*ηmi2))
#     # print("δ : ", δ)
#     # τ = 2*π*K/δ
#     # νm = modes[m][0]*νz
#     # ωd = νm + δ

#     δ = sqrt(abs(4*K*Ωvals[i1]*Ωvals[i2]*ηmi1*ηmi2))
#     τ = 2*π*K/δ
#     νm = modes[m][0]*νz
#     ωd = νm + δ

#     # strength = abs(Ωvals[i1]*Ωvals[i2]*ηmi1*ηmi2)
#     # time_dependence = τ/(ωd-νm) - sin((ωd-νm)*τ)/(ωd-νm)**2
#     # θ = -1/4 * strength * time_dependence

#     t=τ*L

#     U = UMS(t, N, Ωvals, η, ωd, modenums=modenums, modetype=modetype, νz=νz, νr=νr, ϕs=ϕs, ϕm=ϕm, modetrunc=modetrunc)
#     ψf = U * ψ0
    
#     return ψf

# def MSPerfectQubitStateFidelity_FromUnitary(N, m, targets, intensities, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L=1):
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")
#     targetgatedirection = np.sign(modes[m][1][targets[0]]*modes[m][1][targets[1]])
#     ψ0 = tensor(tensor([basis(2,0) for _ in range(N)]), tensor([fock(modetrunc,0) for _ in modenums]))
#     ψf = MSUnitary(N, m, targets, intensities, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L)
#     ρfqbit = ket2dm(ψf).ptrace(list(range(N))) # Trace out modes by tracing out everything except the N qubits
#     idealUMS = (-1j*targetgatedirection*π/4*tensor([sigmax() if i in (i1, i2) else qeye(2) for i in range(N)])).expm()
#     ψ0 = tensor([basis(2,0) for _ in range(N)])
#     idealψfqbit = idealUMS*ψ0
#     return (ρfqbit*ket2dm(idealψfqbit)).tr()

# def MSTracedTargetsFidelity_FromUnitary(N, m, targets, intensities, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L=1):
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")
#     targetgatedirection = np.sign(modes[m][1][targets[0]]*modes[m][1][targets[1]])
#     ψf = MSStateFromUnitary(N, m, targets, intensities, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L)
#     targetsys = ψf.ptrace([i1,i2])
#     idealUMS = (-1j*targetgatedirection*π/4*tensor(sigmax(), sigmax())).expm()
#     ψ0 = tensor(basis(2,0), basis(2,0))
#     idealψfqbit = idealUMS*ψ0
#     return (targetsys*ket2dm(idealψfqbit)).tr()


# def coherent_coeff(α, n):
#     return exp(-abs(α)**2/2)*α**n/sqrt(factorial(n))

# def MSPerfectStateFidelity_Analytical(N, m, targets, Ωvals, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L=1, fidtargets=None):
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")
    
#     i1, i2 = targets
    
#     ηmi1 = η*modes[m][1][i1]
#     ηmi2 = η*modes[m][1][i2]

#     δ = sqrt(abs(4*K*Ωvals[i1]*Ωvals[i2]*ηmi1*ηmi2))
#     τ = 2*π*K/δ
#     νm = modes[m][0]*νz
#     ωd = νm + δ
    
#     if fidtargets == None: fidtargets = targets
#     fidtargetgatedirection = np.sign(modes[m][1][fidtargets[0]]*modes[m][1][fidtargets[1]])
    
#     naturalbasis = list(itertools.product(*[[-1,+1]]*N))
#     innerprod = 0
#     t = τ*L
#     for λ in naturalbasis:
#         displacementproduct = 1
#         for k in modenums:
#             αk = np.sum([(Ωvals[i]*η*modes[k][1][i])/(2*(ωd-modes[k][0]*νz))*(exp(-1j*(ωd-modes[k][0]*νz)*t)-1)*exp(1j*ϕm)*λ[i] for i in range(N)])
#             displacementproduct *= coherent_coeff(αk, 0)
#             #print(λ, k, αk)
#         phaseangle = 0
#         for k in modenums:
#             for j1, j2 in itertools.product(range(N),range(N)):
#                 ηkj1 = η*modes[k][1][j1]
#                 ηkj2 = η*modes[k][1][j2]
#                 νk = modes[k][0]*νz
#                 phaseangle += Ωvals[j1]*Ωvals[j2]*ηkj1*ηkj2*(t/(ωd-νk)-sin((ωd-νk)*t)/(ωd-νk)**2)*λ[j1]*λ[j2]
#         M2phase = exp(-1j/4*phaseangle) # Should that be /2 or /4? Check.
#         #print(λ, M2phase)
#         correctphase = exp(1j*π/4*λ[fidtargets[0]]*λ[fidtargets[1]]*fidtargetgatedirection)
#         innerprod += M2phase*correctphase*displacementproduct/2**(N)
#     fidelity = abs(innerprod)**2
#     return fidelity


# def MSTracedTargetsFidelity_Analytical(N, m, targets, Ωvals, η, modetype, νz, νr, ϕs, ϕm, modenums, modetrunc, K, L=1, fidtargets=None):
#     if modetype == "radial":
#         modes = ict.calcRadialModes(N, νratio=νr/νz)
#     elif modetype == "axial":
#         modes = ict.calcAxialModes(N)
#     else:
#         raise Exception("Only modetype options are 'radial' and 'axial'")
    
#     i1, i2 = targets
    
#     ηmi1 = η*modes[m][1][i1]
#     ηmi2 = η*modes[m][1][i2]

#     δ = sqrt(abs(4*K*Ωvals[i1]*Ωvals[i2]*ηmi1*ηmi2))
#     τ = 2*π*K/δ
#     νm = modes[m][0]*νz
#     ωd = νm + δ
    
#     ρqbit_mat = np.zeros((2**N,2**N), dtype=np.cdouble)
    
#     naturalbasis = list(itertools.product(*[[-1,+1]]*N))
#     innerprod = 0
#     t = τ*L
#     for λ1,λ2 in itertools.product(naturalbasis, naturalbasis): # λ is a N-element list, representing a qubit state in the X basis (-1=|0>, +1=|1>)
#         λ1λ2coeff = 0
#         for γ in list(itertools.product(*[list(range(modetrunc))]*len(modenums))): # γ is a list of N integers, representing a mode state in the number basis
#             displacementproduct = 1/2**(N)
#             for k in modenums:
#                 αλ1k = np.sum([(Ωvals[i]*η*modes[k][1][i])/(2*(ωd-modes[k][0]*νz))*(exp(-1j*(ωd-modes[k][0]*νz)*t)-1)*exp(1j*ϕm)*λ1[i] for i in range(N)])
#                 αλ2k = np.sum([(Ωvals[i]*η*modes[k][1][i])/(2*(ωd-modes[k][0]*νz))*(exp(-1j*(ωd-modes[k][0]*νz)*t)-1)*exp(1j*ϕm)*λ2[i] for i in range(N)])
#                 displacementproduct *= coherent_coeff(αλ1k, γ[modenums.index(k)])*coherent_coeff(αλ2k, γ[modenums.index(k)]).conjugate()
#                 #print(λ, k, αk)
#             λ1λ2coeff += displacementproduct
#         phaseproduct = 1
#         for a, λ in enumerate((λ1, λ2)):
#             phaseangle = 0
#             for k in modenums:
#                 for j1, j2 in itertools.product(range(N),range(N)):
#                     ηkj1 = η*modes[k][1][j1]
#                     ηkj2 = η*modes[k][1][j2]
#                     νk = modes[k][0]*νz
#                     phaseangle += Ωvals[j1]*Ωvals[j2]*ηkj1*ηkj2*(t/(ωd-νk)-sin((ωd-νk)*t)/(ωd-νk)**2)*λ[j1]*λ[j2]
#             phase = exp(-1j/4*phaseangle)
#             if a == 1:
#                 phase = phase.conjugate()
#             phaseproduct *= phase
#         λ1λ2coeff *= phaseproduct
#         λ1_index = int("".join(['0' if λ1[i]==-1 else '1' for i in range(N)]),2)
#         λ2_index = int("".join(['0' if λ2[i]==-1 else '1' for i in range(N)]),2)
#         ρqbit_mat[λ1_index, λ2_index] = λ1λ2coeff
        
#     if fidtargets == None: fidtargets = targets
#     fidtargetgatedirection = np.sign(modes[m][1][fidtargets[0]]*modes[m][1][fidtargets[1]])
    
#     ρqbit = qtp.Qobj(ρqbit_mat, dims=[[2]*N,[2]*N])
#     #print(ρqbit.tr())
#     ρred = ρqbit.ptrace(fidtargets)
#     #print(ρred.tr())
#     idealUMS = (-1j*fidtargetgatedirection*π/4*tensor(sigmax(), sigmax())).expm()
#     ψ0 = tensor(basis(2,0), basis(2,0))
#     idealψfqbit = idealUMS*ψ0
#     eigstates = σ(π/2+ϕs).eigenstates()
#     change_of_basis = qtp.Qobj(np.c_[eigstates[1][0].data.A, eigstates[1][1].data.A],dims=[[2],[2]])
#     idealψfqbit_zbasis = tensor(change_of_basis,change_of_basis)*idealψfqbit
#     #print(ρred)
#     #print(ket2dm(idealψfqbit))
#     #print(ket2dm(idealψfqbit_zbasis))
#     return (ρred*ket2dm(idealψfqbit_zbasis)).tr()