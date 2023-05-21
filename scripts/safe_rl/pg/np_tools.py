import numpy as np
def mpi_statistics_scalar(x):
    '''
    计算出mean/std 和合适的min/max
    x: 一个装满标量的数组
    with_min_and_max: 如果为真，则额外返回max和min
    '''
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x) # 通过多线程并行计算sum，加快执行速度
    mean = global_sum / global_n
    global_sum_sq = np.sum(np.sum((x-mean)**2))
    std = np.sqrt(global_sum_sq / global_n)
    return mean, std

def norm(x):
    mean, std = mpi_statistics_scalar(x)
    x = (x-mean) / std
    return x