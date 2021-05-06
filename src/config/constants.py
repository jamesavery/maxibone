implant_threshold = 5
global_vmin, global_vmax = -4,12
implant_threshold_byte = int(((implant_threshold-global_vmin)/(global_vmax-global_vmin))*256 + 1)
