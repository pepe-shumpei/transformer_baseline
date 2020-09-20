def lr_schedule(step):
    step = step + 1
    a = 512 ** (-0.5)
    b = min([step ** (-0.5), step*4000**(-1.5)])
    return a * b 