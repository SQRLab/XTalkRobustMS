import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sys import argv
from glob import glob
import os
import pandas as pd


def makeIonArray(prevArray, numIons, numShots, numPhases, numStates):
    ionArray = np.zeros(shape = (numIons, numShots, numPhases))
    for i in range(numPhases):
        for k in range(numStates):
            for j in range(numShots):
                if prevArray[i,k,j] == 1:
                    if k == 1:
                        ionArray[2,j,i] = 1
                    elif k == 2:
                        ionArray[1,j,i] = 1
                    elif k == 3:
                        ionArray[1,j,i] = 1
                        ionArray[2,j,i] = 1
                    elif k == 4:
                        ionArray[0,j,i] = 1
                    elif k == 5:
                        ionArray[0,j,i] = 1
                        ionArray[2,j,i] = 1
                    elif k == 6:
                        ionArray[0,j,i] = 1
                        ionArray[1,j,i] = 1
                    elif k == 7:
                        ionArray[0,j,i] = 1
                        ionArray[1,j,i] = 1
                        ionArray[2,j,i] = 1
    return ionArray

def getProbabilities(ionArray):
    numIons = ionArray.shape[0]
    numShots = ionArray.shape[1]
    numDataPoints = ionArray.shape[2]
    pt1t2 = np.zeros(numDataPoints)
    pt1n = np.zeros(numDataPoints)
    pt2n = np.zeros(numDataPoints)
    for i in range(numDataPoints):
        for j in range(numShots):
            if ionArray[0,j,i] == ionArray[2,j,i]:
                pt1t2[i] += 1/numShots
            else:
                pt1t2[i] -= 1/numShots
            if ionArray[0,j,i] == ionArray[1,j,i]:
                pt1n[i] += 1/numShots
            else:
                pt1n[i] -= 1/numShots
            if ionArray[1,j,i] == ionArray[2,j,i]:
                pt2n[i] += 1/numShots
            else:
                pt2n[i] -= 1/numShots

    return [pt1t2, pt1n, pt2n]

file_base = "MS_Gate_Parity_250us_Mode0_15kHz_"
file_end = "percentxtalk"

# Construct the full file path
#file_path = os.path.join(folder_path, file_name)

# For New Parity Curves
variable_names = [
    "timeTickFirst", "timeTickLast", "x", "q0_detect", "q0_detect_raw", "q0_detect_bottom", "q0_detect_top",
    "q0_cool", "q0_cool_raw", "q0_cool_bottom", "q0_cool_top", "q1_detect", "q1_detect_raw", "q1_detect_bottom",
    "q1_detect_top", "q1_cool", "q1_cool_raw", "q1_cool_bottom", "q1_cool_top", "qn1_detect", "qn1_detect_raw",
    "qn1_detect_bottom", "qn1_detect_top", "qn1_cool", "qn1_cool_raw", "qn1_cool_bottom", "qn1_cool_top",
    "q2_detect", "q2_detect_raw", "q2_detect_bottom", "q2_detect_top", "qn2_detect", "qn2_detect_raw",
    "qn2_detect_bottom", "qn2_detect_top", "ZeroBright", "ZeroBright_raw", "ZeroBright_bottom", "ZeroBright_top",
    "OneBright", "OneBright_raw", "OneBright_bottom", "OneBright_top", "TwoBright", "TwoBright_raw",
    "TwoBright_bottom", "TwoBright_top", "Parity", "Parity_raw", "Parity_bottom", "Parity_top", "chain_as_int",
    "chain_as_int_raw", "chain_as_int_bottom", "chain_as_int_top"
]

# For Data Set 1
# variable_names = [
#     "timeTickFirst", "timeTickLast", "x", "q0_detect", "q0_detect_raw", "q0_detect_bottom", "q0_detect_top",
#     "q0_cool", "q0_cool_raw", "q0_cool_bottom", "q0_cool_top", "q1_detect", "q1_detect_raw", "q1_detect_bottom",
#     "q1_detect_top", "q1_cool", "q1_cool_raw", "q1_cool_bottom", "q1_cool_top", "qn1_detect", "qn1_detect_raw",
#     "qn1_detect_bottom", "qn1_detect_top", "qn1_cool", "qn1_cool_raw", "qn1_cool_bottom", "qn1_cool_top",
#     "q2_detect", "q2_detect_raw", "q2_detect_bottom", "q2_detect_top", "qn2_detect", "qn2_detect_raw",
#     "qn2_detect_bottom", "qn2_detect_top", "TwoBright", "TwoBright_raw", "TwoBright_bottom", "TwoBright_top",
#     "OneBright", "OneBright_raw", "OneBright_bottom", "OneBright_top", "ZeroBright", "ZeroBright_raw",
#     "ZeroBright_bottom", "ZeroBright_top", "Elliptical_PD_Detect", "Elliptical_PD_Detect_raw",
#     "Elliptical_PD_Detect_bottom", "Elliptical_PD_Detect_top", "Elliptical_PD_Cool", "Elliptical_PD_Cool_raw",
#     "Elliptical_PD_Cool_bottom", "Elliptical_PD_Cool_top", "NoiseEater355_PD_Cool", "NoiseEater355_PD_Cool_raw",
#     "q2_cool", "q2_cool_raw", "q2_cool_bottom", "q2_cool_top", "qn2_cool", "qn2_cool_raw"
# ]

data_dict = {}
for name in variable_names:
    data_dict[name] = ""

fidelity_target = np.zeros(6)
fidelity_XTalk = np.zeros(6)
error_target = np.zeros(6)
error_XTalk = np.zeros(6)

xtalks = [0, 10, 20, 30, 40, 50]

for xtalk in xtalks:
    prob = np.load(file_base+str(xtalk)+file_end+"_Raw_prob_data.npy")
    shot = np.load(file_base+str(xtalk)+file_end+"_Raw_shot_data.npy")
    ionShotArray = makeIonArray(shot, 3, 100, 67, 8)
    ionProbArray = getProbabilities(ionShotArray)
    file_name = file_base+str(xtalk)+file_end

    try:
        if len(argv) > 1:
            file_path = argv[1]

        # Open the file with the correct encoding
        with open(file_name, 'r', encoding='latin1') as file:
            # Load data from file with mixed data types
            dat_1 = np.genfromtxt(file, delimiter='\t', skip_header=252, dtype=None, encoding='latin1')
            
            len_dat1 = len(dat_1)

            # Fills in the dictionary with the data
            for i, name in enumerate(variable_names):
                col = []
                for j in range(dat_1.shape[0]):
                    col.append(dat_1[j][i])
                data_dict[name] = col

    except IOError:
        print("** Error reading specified file - aborting **")
        exit()
    except UnicodeDecodeError as e:
        print(f"** Unicode decode error: {e} - aborting **")
        exit()

    parity = np.zeros(len(data_dict["TwoBright"]))

    for i in range(len(data_dict["TwoBright"])):
        parity[i] = data_dict["TwoBright"][i] + data_dict["ZeroBright"][i] - data_dict["OneBright"][i]

    phase = np.zeros(len(data_dict["x"]))
    for i in range(len(data_dict["x"])):
        phase[i] = data_dict["x"][i]*np.pi/180

    df = pd.DataFrame({'phase': phase, 'parity': parity, 'rawt1t2' : ionProbArray[0], \
                    'rawt1n' : ionProbArray[1], 'rawt2n' : ionProbArray[2]})
    df = df.sort_values('phase')

    # Fit a sine function to the parity data
    def sin_func(x, A, B, C, D):
        y = A * np.sin(B * x + C) + D
        return y

    guess = [.95, 2, 0, 0]  # Initial guess for the parameters A, B, and C
    parameters, covariance = curve_fit(sin_func, df['phase'],df['parity'], p0=guess)
    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    fit_D = parameters[3]

    target_uncertainty = np.sqrt(covariance[0, 0])

    parameters, covariance = curve_fit(sin_func, df['phase'],df['rawt1n'], p0=guess)
    fit_XA = parameters[0]
    fit_XB = parameters[1]
    fit_XC = parameters[2]
    fit_XD = parameters[3]

    xtalk_uncertainty = np.sqrt(covariance[0, 0])
    
    fit_sine = sin_func(df['phase'], fit_A, fit_B, fit_C, fit_D)
    
    fidelity_target[int(xtalk/10)] = np.abs(fit_A)
    fidelity_XTalk[int(xtalk/10)] = np.abs(fit_XA)
    error_target[int(xtalk/10)] = target_uncertainty
    error_XTalk[int(xtalk/10)] = xtalk_uncertainty

plt.plot(xtalks, fidelity_target, label="Target Fidelity")
plt.plot(xtalks, fidelity_XTalk, label="Cross Talk Fidelity")
plt.xlabel("XTalk")
plt.ylabel("Fidelity")
plt.title(f"{file_base}_0-50")
plt.legend()
plt.show()

save_name = file_base+"_0-50"

np.save(save_name+"_Target_Fidelity.npy", fidelity_target)
np.save(save_name+"_XTalk_Fidelity.npy", fidelity_XTalk)
np.save(save_name+"_Target_Error.npy", error_target)
np.save(save_name+"_XTalk_Error.npy", error_XTalk)
