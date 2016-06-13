import numpy as np

def extend_true(x, LEFT=0, RIGHT=0):
    '''If a[x]=True, it will make a[x-LEFT+1:x]=True and a[x,x+RIGHT+1]=True as well.
    '''
    idx = np.where(x)[0]
    for i in range(1, LEFT+1):
        ext_ind = idx - i
        ext_ind[ext_ind<0] = 0
        x[ext_ind] = True
    for i in range(1, RIGHT+1):
        ext_ind = idx + i
        ext_ind[ext_ind>len(x)-1] = len(x)-1
        x[ext_ind] = True
    return x

def skip_outside_frame_start_to_end(bool_arr, FRAME_START, FRAME_END):
    skip = np.zeros(bool_arr.shape, bool)
    skip[:, FRAME_START:FRAME_END] = True
    bool_arr *= skip
    return bool_arr
