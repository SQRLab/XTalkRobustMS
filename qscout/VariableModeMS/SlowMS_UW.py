##ADAPTED FROM SANDIA gate_MS_theta code

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
from jaqalpaw.utilities.datatypes import Parallel
import numpy as np

from QSCOUTBuiltins_live import QSCOUTBuiltins, GLOBAL_BEAM

## Constant Values.
# Tone mask binary values for convenience.
tone0 = 0b01
tone1 = 0b10
no_tones = 0b00
both_tones = 0b11

MS_red_amp_list: list = [(40,40)]*10 # N choose 2 elements.
MS_blue_amp_list: list = [(44,44)]*10
MS_framerot: list = [0.0]*10
MS_framerot_IA: list = [0.0]*10
    
MS_sb_idx_list: list = [] # [-1]*10
MS_intxn_sign: list = [1]

def gate_MS_slowUW(self, channel1, channel2, axis_rad=0, angle=np.pi/2,
                    mode=0, frame_forwarding=both_tones):
        
        if self.qubit_mapping[channel1] > self.qubit_mapping[channel2]: 
            temp = channel1
            channel1 = channel2 
            channel2 = temp             
                
        target_idx = self.ms_target_idx(channel1,channel2)
        
        ## Calculate relevant values from the calibrated parameters.
        # Use global beam as the lower leg of the Raman transition.        
        global_beam_frequency =  discretize_frequency(self.ia_center_frequency) - discretize_frequency(self.MS_adjusted_carrier_splitting)

        # Convert detuning knots to actual RF drive frequencies. Blue=fm0, Red=fm1
        freq_fm0 = discretize_frequency(self.ia_center_frequency) + discretize_frequency(self.MS_delta) + discretize_frequency(higher_motional_mode_frequencies[mode])
        
        freq_fm1 = discretize_frequency(self.ia_center_frequency) - discretize_frequency(self.MS_delta) - discretize_frequency(higher_motional_mode_frequencies[mode])
        
        
        #Global beam AOM distortion (note: this assumes the global AOM power input is 100 for a pi/2 MS gate (XX)) 
        global_amp_cal = self.global_amp_cal(angle, self.Counter_Amp0_M, self.MS_global_amp)
        global_amp = self.MS_global_amp
        
        framerot_input = tuple(np.linspace(0, 1, len(amp_scale))*self.MS_framerot)
        framerot_app = 0
        amp_scale = np.array([1,1])
        
        global_amp = tuple(self.MS_global_amp*global_amp_cal*amp_scale)
        
            
        #Convert Jaqal input axis from radians to degrees as the hardware requires degrees
        #Note: for integration with single qubit gates, this defaults to zero and the MS gate phase is determined by an external set of gates (see third exemplar)
        axis = axis_rad*180/np.pi
        
        #If requested MS gate angle is negative, then one ion in the pair will be performed 180 degrees out of phase to get the desired X,-X interaction to create a negative MS gate angle
        phase_add = 180 if angle < 0 else 0
        if self.MS_intxn_sign[target_idx] < 0:
            phase_add += 180

        listtoReturn = [PulseData(GLOBAL_BEAM, self.MS_pulse_duration,
                                  freq0=global_beam_frequency,
                                  freq1=self.global_center_frequency, # This tone is not used, frequency doesn't matter.
                                  amp0=global_amp,
                                  amp1=0,
                                  phase0=0,
                                  phase1=0,
                                  sync_mask=both_tones,
                                  fb_enable_mask=tone0),
                        PulseData(channel1, self.MS_pulse_duration,
                                  freq0=freq_fm0,
                                  freq1=freq_fm1,
                                  amp0=tuple(self.MS_blue_amp_list[target_idx][amp_index0]*amp_scale),
                                  amp1=tuple(self.MS_red_amp_list[target_idx][amp_index0]*amp_scale),
                                  framerot0 = framerot_input,
                                  framerot1 = framerot_input,
                                  fwd_frame0_mask = frame_forwarding,
                                  fwd_frame1_mask = both_tones,
                                  apply_at_end_mask=framerot_app,
                                  phase0=self.MS_phi + axis,
                                  phase1=self.MS_phi + axis,
                                  sync_mask=both_tones,
                                  fb_enable_mask=no_tones
                                  ),
                        PulseData(channel2, self.MS_pulse_duration,
                                  freq0=freq_fm0,
                                  freq1=freq_fm1,
                                  amp0=tuple(self.MS_blue_amp_list[target_idx][amp_index1]*amp_scale),
                                  amp1=tuple(self.MS_red_amp_list[target_idx][amp_index1]*amp_scale),
                                  phase0=self.MS_phi + axis + phase_add,
                                  phase1=self.MS_phi + axis + phase_add,
                                  framerot0 = framerot_input,
                                  framerot1 = framerot_input,
                                  fwd_frame0_mask = frame_forwarding,
                                  fwd_frame1_mask = both_tones,
                                  apply_at_end_mask=framerot_app,
                                  sync_mask=both_tones,
                                  enable_mask= both_tones if singleion > 0 else no_tones,
                                  fb_enable_mask=no_tones
                                  )]
        return listtoReturn