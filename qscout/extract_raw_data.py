import numpy as np 
import matplotlib.pyplot as plt 
import os
import re
from collections import Counter

# Dictionary for converting names of evaluations for qscout project
RAW_EVALS = {'cool_count0':256, 'cool_count1':257, 'cool_count2':258, 'cool_count3':259, # Cool Counters
             'det_count0':0, 'det_count1':1, 'det_count2':2, 'det_count3':3, # Cool Counters
            }

# Definitions to parse standard data sets and raw data sets
def parse_raw(file_path, skip_lines=None, end_line=None, get_counters=['check_3'], fast_parse=True):
    """ Parses ioncontrol 'raw' format datafiles. 
    Each line in the raw file is a single scanpoint. I think they're just recorded in the order they come in - MC.
    Returns: data : list of lists with format   
        0: list of scan variable values
        1: list of arrays (one for each scan point) containing counters[0] values
        ...
        len(counters): list of arrays (one for each scan point) containing counters[len(counters)-1] values
    """
    data = [[] for i in range(len(get_counters)+1)]
    scan_var = []
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            i += 1
            if skip_lines is not None and i <= skip_lines:
                continue
            elif end_line is not None and i > end_line:
                break
            if fast_parse:
                """ Manually parse through each line, looking for particular start flags 
                that are associated with each counter and the scan variable. This allows us to 
                parse only necessary parts of the line. MC 2022-03-17"""
                scan_var_flag = "null, null," # String that precedes the scan variable entry.
                scan_var_start_idx = line.find(scan_var_flag)+len(scan_var_flag)
                scan_var_end_idx = line.find(",", scan_var_start_idx)
                scan_var.append(float(line[scan_var_start_idx:scan_var_end_idx]))
                for j in range(len(get_counters)):
                    flag = r'"%d": ' % RAW_EVALS[get_counters[j]]
                    start_idx = line.find(flag)+len(flag)
                    end_idx = line.find("]", start_idx)+1
                    counts = eval(line[start_idx:end_idx])
                    data[j+1].append(np.array(counts))
            else:
                """ Old way of doing parsing that evaluates the entire line and then selects
                relevant portions to return. Much slower, but tested extensively. """
                line = line.replace("true", "True")
                line = line.replace("false", "False")
                line = line.replace("null", "None")
                line = eval(line)
                scan_var.append(line[3])
                counters = line[0]  # dictionary for all counters 
                results = line[9] # dictionary for all the results
                all_counters = {**counters, **results}
                if True: # TODO: add some filter conditions here.  
                    for j in range(len(get_counters)):
                        if get_counters[j] in RAW_EVALS.keys():
                            data[j+1].append(np.array(all_counters[str(RAW_EVALS[get_counters[j]])]))
                        else:
                            data[j+1].append(np.array(all_counters[str(get_counters[j])]))
    data[0] = scan_var
    return data

#Functions to convert raw detect counts into two-qubit states
def threshold_detect_counts(array):
    """ Function to threshold detection counts, converting an array of shot-by-shot single qubit detection
    counts to shot-by-shot single qubit state assignments. As the threshold in IonControl for the fiber array is set to 1,
    we set the threshold here to 1 as well. Counts of 0 or 1 are assumed to be a dark state, |0>, while all counts
    of 2 or more are assumed to be the bright state, |1>. This function looks through the shot-by-shot detection count array,
    and returns the same-sized array with ones or zeros. Ones correspond to |1>, zeros correspond to |0>."""
    thold = 1
    thresholded_array = []
    for element in array:
        if element > thold:
            element = 1
            thresholded_array.append(element)
        else:
            element = 0
            thresholded_array.append(element)
    return thresholded_array

def three_qubit_states_shot(qn1,q0,q1):
    """Function converts two sets of single-qubit shot-by-shot states to a single array of shot-by-shot two qubit states.
    Note the naming convention of the qubit states matches that of IonControl - "dd" means dark-dark or |00>, likewise "bd" is |10>.
    These are what are known in JaqalPaq as the "str" convention, i.e. when you call for probability_by_str. However, we output the
    results as the "int" convention ordering. That is state |10> is q0 = 1, q1 = 0, and so is stored as an integer 01. Likewise, state |01> means
    q0 = 0 and q1 = 1, and will be stored as the binary integer 10."""
    q0thold = threshold_detect_counts(q0)
    q1thold = threshold_detect_counts(q1)
    qn1thold = threshold_detect_counts(qn1)
    shots = len(q0thold)
    ddd = np.zeros(shots)
    bdd = np.zeros(shots)
    dbd = np.zeros(shots)
    bbd = np.zeros(shots)
    ddb = np.zeros(shots)
    bdb = np.zeros(shots)
    dbb = np.zeros(shots)
    bbb = np.zeros(shots)

    for iii in range(shots):
        if q0thold[iii] == 1:
            if qn1thold[iii] == 1:
                if q1thold[iii] == 1:
                    bbb[iii] = 1
                else:
                    bbd[iii] = 1
            elif q1thold[iii] == 1:
                dbb[iii] = 1
            else:
                dbd[iii] = 1
        elif q1thold[iii] == 1:
            if qn1thold[iii] == 1:
                bdb[iii] =1
            else:
                ddb[iii] = 1
        else:
            if qn1thold[iii] == 1:
                bdd[iii] = 1
            else:
                ddd[iii] = 1
    print([ddd, ddb, dbd, dbb, bdd, bdb, bbd, bbb])
    return [ddd, ddb, dbd, dbb, bdd, bdb, bbd, bbb]

def three_qubit_states_prob(qn1, q0, q1):
    """This function just outputs the overall probabilities exported in the probability_by_int convetion"""
    res = three_qubit_states_shot(qn1, q0, q1)
    shots = len(q0)
    return [sum(res[0])/shots, sum(res[1])/shots, sum(res[2])/shots, sum(res[3])/shots, sum(res[4])/shots, sum(res[5])/shots, sum(res[6])/shots, sum(res[7])/shots]

#File location
#root = "MS_Gate_Parity_VariableMode_Mode"
root = "MS_Gate_Parity_250us_Mode"
end = "_20kHz_0percentxtalk_Raw"

#Create empty arrays to fill
files = []

#Append the necessary files into the arrays.
# for ii in range(0,3):
#    files.append(root+str(ii)+end)
files.append("MS_Gate_Parity_250us_Mode2_20kHz_50percentxtalk_Raw")

phase_value = 67
num_states = 8
shot_value = 100
#Create empty arrays to fill with shot-by-shot two-qubit state data
shot_data = np.zeros(shape = (phase_value,num_states,shot_value))
prob_data = np.zeros(shape = (phase_value,num_states))

#Fill up those arrays
num_files = len(files)
for ii in range(num_files):
    qn1 = parse_raw(files[ii], get_counters = ['det_count3'])[1]
    q0 = parse_raw(files[ii],get_counters = ['det_count1'])[1]
    q1 = parse_raw(files[ii],get_counters = ['det_count2'])[1]
    for jj in range(0,phase_value):
        prob_data[jj] = three_qubit_states_prob(qn1[jj], q0[jj],q1[jj])
        shot_data[jj] = three_qubit_states_shot(qn1[jj], q0[jj],q1[jj])
        # shot_data[jj] = three_qubit_states_shot(qn1[jj*2], q0[jj*2],q1[jj*2])
        # prob_data[jj] = three_qubit_states_prob(qn1[jj*2], q0[jj*2],q1[jj*2])
    np.save(files[ii]+"_shot_data",shot_data)
    np.save(files[ii]+"_prob_data",prob_data)

    

#Save as numpy arrays
#np.save(root+"shot_data_test",shot_data)
#np.save(root+"prob_data_test",prob_data)



