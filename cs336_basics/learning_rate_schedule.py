import math

def get_lr_cosine_schedule(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t / T_w * lr_max
    elif T_w <= t and t <= T_c:
        return lr_min + (1 / 2) * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
