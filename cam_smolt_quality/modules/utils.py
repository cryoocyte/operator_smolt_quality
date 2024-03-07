import numpy as np


def eSFR (row):
    w = row['open_weight']
    t = row['degree_days']
    yf = (.2735797591)+(-.0720137809*t)+(.0187408253*t**2)+(-.0008145337*t**3)
    y0 = (-.79303459)+(.43059382*t)+(-.01471246*t**2)
    log_alpha = (-7.8284505676)+(.3748824960*t)+(-.0301640851*t**2)+(.0006516355*t**3)
    return (yf - (yf-y0)*np.exp(-np.exp(log_alpha)*w))