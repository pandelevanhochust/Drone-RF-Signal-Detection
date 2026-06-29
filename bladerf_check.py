from bladerf import _bladerf

import numpy as np
import matplotlib.pyplot as plt

sdr = _bladerf.BladeRF()

print("Device info:", _bladerf.get_device_list()[0])
print("libbladeRF version:", _bladerf.version()) # v2.5.0
print("Firmware version:", sdr.get_fw_version()) # v2.4.0
print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # give it a 0 or 1
print("sample_rate_range:", rx_ch.sample_rate_range)
print("bandwidth_range:", rx_ch.bandwidth_range)
print("frequency_range:", rx_ch.frequency_range)
print("gain_modes:", rx_ch.gain_modes)
print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # ch 0 or 1