from numba import njit, prange
from tqdm import tqdm
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

@njit(
    cache = True,
    fastmath = True
)
def doesDiverge(c: np.complex128, maxIter: int, divergenceLimit: int) -> bool:
    z = np.cdouble(c.real + c.imag * 1j)

    for _ in range(maxIter):
        z = c ** z
        if np.abs(z) > divergenceLimit:
            return True
    
    return False

@njit(
    cache = True,
    fastmath = True,
    parallel = True
)
def getDivergenceMap(
    screenWidth: int,
    screenHeight: float,
    x: float,
    y: float,
    epsX: float,
    epsY: float,
    maxIter: int,
    divergenceLimit: int
):
    xAxis = np.linspace(x - epsX, x + epsX, screenWidth)
    yAxis = np.linspace(y - epsY, y + epsY, screenHeight)

    divergenceMap = np.zeros(
        (screenHeight, screenWidth),
        dtype = np.bool_
    )

    for i in prange(screenHeight):
        for j in prange(screenWidth):
            c = np.cdouble(xAxis[j] + yAxis[i] * 1j)

            divergenceMap[i, j] = doesDiverge(c, maxIter, divergenceLimit)

    return divergenceMap

def showFrame(frame: np.ndarray, x: float, y: float, epsX: float, epsY):
    plt.imshow(frame, extent = [x - epsX, x + epsX, y - epsY, y + epsY], origin='lower')
    plt.axis('off')
    plt.show()

def saveFrame(
    frame: np.ndarray,
    screenWidth: int,
    screenHeight: float,
    x: float,
    y: float,
    eps: float,
    maxIter: int,
):
    fileName = f"{os.getcwd()}/images/ptf_{screenWidth}_{screenHeight}_{x}_{y}_{eps}_{maxIter}.png"

    plt.imsave(fileName, frame)

color = 0
cmap = plt.get_cmap("hsv")

print(cmap(6))

def getFrame(
    screenWidth: int,
    screenHeight: float,
    x: float,
    y: float,
    eps: float,
    maxIter: int,
    divergenceLimit: int,
    show: bool = False,
    save: bool = False
):
    global color, cmap

    epsX, epsY = eps, eps * (screenHeight / screenWidth)
    frame = np.full(
        shape = ( screenHeight, screenWidth, 3 ),
        fill_value = (0, 0, 0),
        dtype = np.uint8
    )
    color += 0.125 / 16
    color = color % 1

    divergenceMap = getDivergenceMap(
        screenWidth = screenWidth,
        screenHeight = screenHeight,
        x = x,
        y = y,
        epsX = epsX,
        epsY = epsY,
        maxIter = maxIter,
        divergenceLimit = divergenceLimit
    )


    for i in range(screenHeight):
        for j in range(screenWidth):
            if divergenceMap[i, j]:
                frame[i, j] = tuple(int(i * 255) for i in cmap(color)[:3])
            else:
                frame[i, j] = ( 0, 0, 0 )
    if save:
        saveFrame(frame, screenWidth, screenHeight, x, y, eps, maxIter)
    if show:
        showFrame(frame, x, y, epsX, epsY)

    return frame

def saveStaticZoomingVideo(
    fps: int,
    duration: float,
    screenWidth: int,
    screenHeight: float,
    x: float,
    y: float,
    fromEps: float,
    toEps: float,
    maxIter: int,
    divergenceLimit: int,
    easeing: Callable = lambda t: t
):
    numberOfFrames = int(fps * duration)

    video = cv2.VideoWriter(
        filename = f"{os.getcwd()}/videos/ptf_{screenWidth}_{screenHeight}_{x}_{y}_{fromEps}_{toEps}_{maxIter}_{fps}fps.mp4",
        fourcc = cv2.VideoWriter_fourcc(*"MJPG"),
        fps = fps,
        frameSize = (screenWidth, screenHeight)
    )

    epsStep = (toEps / fromEps) ** (1 / numberOfFrames)

    for i in tqdm(range(numberOfFrames), leave = True):
        t = i / (numberOfFrames - 1)
        eps = fromEps * (epsStep ** (easeing(t) * numberOfFrames))
        frame = getFrame(
            screenWidth = screenWidth,
            screenHeight = screenHeight,
            x = x,
            y = y,
            eps = eps,
            maxIter = maxIter,
            divergenceLimit = divergenceLimit
        )
        video.write(frame)
    
    video.release()
    cv2.destroyAllWindows()

