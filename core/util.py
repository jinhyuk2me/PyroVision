import time

def dyn_sleep(s_time, max_time):
    d_time = time.time() - s_time
    if d_time < max_time:
        time.sleep(max_time-d_time)