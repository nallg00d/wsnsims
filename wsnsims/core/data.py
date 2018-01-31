import numpy as np

data_memo = {}


def segment_volume(src, dst, env):
    if (src, dst) in data_memo:
        return data_memo[(src, dst)]
    
    # Commenting out for testing, put back when doing acutal sim
    #size = np.random.normal(env.isdva, env.isdvsd)
    size = src.volume + dst.volume
    # size *= env.isdva.units
    data_memo[(src, dst)] = size
    return size
