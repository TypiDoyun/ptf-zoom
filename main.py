import numpy as np
from utils.ptf import *
import os

x = -4.115613745471
y = -9e-12
resolution = 960
# 4K 3840x2160
# QHD 2560x1440
# Full HD 1920x1080
# HD+ 1600x900
# HD 1280x720
# qHD 960x540

eps = 5e0
maxIter = 128
divergenceLimit = 1e+10

screenRatio = 9 / 16
screenWidth, screenHeight = resolution, int(resolution * screenRatio)

# getFrame(
#     screenWidth = screenWidth,
#     screenHeight = screenHeight,
#     x = x,
#     y = y,
#     eps = eps / 10000000000,
#     maxIter = maxIter,
#     divergenceLimit = divergenceLimit,
#     show = True
# )

frame = saveStaticZoomingVideo(
    fps = 60,
    duration = 10, 
    screenWidth = screenWidth,
    screenHeight = screenHeight,
    x = x,
    y = y,
    fromEps = eps,
    toEps = eps / 10000000000,
    maxIter = maxIter,
    divergenceLimit = divergenceLimit
)