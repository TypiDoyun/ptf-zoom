import numpy as np
from utils.ptf import *

x = -4.115613745471
y = -9e-12
resolution = 1280
# 4K 3840x2160
# QHD 2560x1440
# Full HD 1920x1080
# HD+ 1600x900
# HD 1280x720
# qHD 960x540

eps = 5e0
maxIter = 128
divergenceLimit = 1e+10

frameRatio = 9 / 16
frameWidth, frameHeight = resolution, int(resolution * frameRatio)

ptf = PTF(
    frameWidth = frameWidth,
    frameRatio = frameRatio,
    maxIter = maxIter,
    divergenceLimit = divergenceLimit
)

frame = ptf.saveStaticZoomVideo(
    fps = 30,
    runningTime = 10,
    x = x,
    y = y,
    startEps = eps,
    endEps = eps / 10000000
)