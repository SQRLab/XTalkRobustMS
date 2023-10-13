""" Variable Mode MS Gate
    Used to generate MS gates using a specified mode.
"""

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
import numpy as np

from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM, both_tones

class VariableModeMS(QSCOUTBuiltins, HelperFunctions):
    # This class inherits both QSCOUTBuiltins and HelperFunctions

    def gate_VariableModeMS(self, channel1, channel2, mode, global_duration=-1e6):
        """VariableModeMS: Generate a Mølmer-Sørensen gate to entangle ions on
            channel1 and channel2 using the specified mode. This code currently
            generates a square pulse on both the global beam and the individual
            addressing beams."""
        ## Note, tuple input to PulseData indicates a spline. List is instant jumps.

        # Use global beam as the lower leg of the Raman transition.
        global_beam_frequency = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)

        blue_frequency = discretize_frequency(self.ia_center_frequency) \
                + discretize_frequency(higher_motional_mode_frequencies[mode]) \
                + discretize_frequency(self.MS_delta)

        red_frequency = discretize_frequency(self.ia_center_frequency) \
                - discretize_frequency(higher_motional_mode_frequencies[mode]) \
                - discretize_frequency(self.MS_delta)

        listtoReturn = [
            PulseData(
                GLOBAL_BEAM,
                self.MS_pulse_duration,
                freq0=self.global_beam_frequency,
                amp0=tuple(np.array([1,1])*self.amp0_counterprop),
                phase0=0,
                phase1=0,
                sync_mask=0b11,
                fb_enable_mask=0b01,
            ),
            PulseData(
                channel1,
                self.MS_pulse_duration,
                freq0=blue_frequency,
                freq1=red_frequency,
                amp0=tuple(np.array([1, 1])*self.MS_blue_amp_list[channel1]),
                amp1=tuple(np.array([1, 1])*self.MS_red_amp_list[channel1]),
                phase0=self.MS_phi + axis,
                phase1=self.MS_phi + axis,
                framerot0=self.MS_framerot,
                apply_at_end_mask=0b01,
                fwd_frame0_mask=both_tones,
                sync_mask=0b11,
                fb_enable_mask=0b00,
            ), 
            PulseData(
                channel2,
                self.MS_pulse_duration,
                freq0=blue_frequency,
                freq1=red_frequency,
                amp0=tuple(self.MS_blue_amp_list[channel2] * np.array([1,1])),
                amp1=tuple(self.MS_red_amp_list[channel2] * np.array([1,1])),
                phase0=self.MS_phi + axis,
                phase1=self.MS_phi + axis,
                framerot0=self.MS_framerot,
                apply_at_end_mask=0b01,
                fwd_frame0_mask=both_tones,
                sync_mask=0b11,
                fb_enable_mask=0b00,
            )
        )
        return listtoReturn


class jaqal_pulses:
    GatePulses = VariableModeMS
