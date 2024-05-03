import numpy as np
  
def read_timestamps_file(path_to_timestamps):
    timestamps = []
    if path_to_timestamps.split('/')[-1] == 'timestamps.txt':
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[1]))
        f.close()
    else:
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[0]))
        f.close()
    t0 = timestamps[0]
    timestamps = list(np.array(timestamps) - t0)
    return timestamps

