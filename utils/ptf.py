from numba import njit, prange
from tqdm import tqdm
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class PTFImage:
    def __init__(
        self,
        frame: np.ndarray[np.int8],
        frameWidth: int,
        frameHeight: int,
        x: float,
        y: float,
        eps: float
    ):
        self.frame = frame
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.x = x
        self.y = y
        self.eps = eps

    def show(
            self,
            frame: np.ndarray
        ):
        epsX, epsY = self.eps, self.eps * (self.frameHeight / self.frameWidth)
        plt.imshow(frame, extent = [x - epsX, x + epsX, y - epsY, y + epsY], origin='lower')
        plt.axis('off')
        plt.show()
    
    def save(
        self
    ):
        fileName = f"{os.getcwd()}/images/ptf_{self.screenWidth}_{self.screenHeight}_{self.x}_{self.y}_{self.eps}.png"
        plt.imsave(fileName, self.frame)


@njit(
    fastmath = True
)
def doesDiverge(
    c: np.complex128,
    maxIter: int,
    divergenceLimit: int
) -> bool:
    z = np.cdouble(c.real + c.imag * 1j)

    for _ in range(maxIter):
        z = c ** z
        if np.abs(z) > divergenceLimit:
            return True
    
    return False


@njit(
    fastmath = True,
    parallel = True
)
def plotFrame(
    frame: np.ndarray[np.int8],
    frameWidth: int,
    frameHeight: int,
    x: float,
    y: float,
    eps: float,
    maxIter: int,
    divergenceLimit: int
):
    frameRatio = frameHeight / frameWidth
    epsX, epsY = eps, eps * frameRatio
    xAxis = np.linspace(x - epsX, x + epsX, frameWidth)
    yAxis = np.linspace(y - epsY, y + epsY, frameHeight)

    for i in prange(frameHeight):
        line = frame[i]
        for j in prange(frameWidth):
            c = np.cdouble(xAxis[j] + yAxis[i] * 1j)

            if doesDiverge(c, maxIter, divergenceLimit):
                line[j] = ( 255, 255, 255 )
            else:
                line[j] = ( 0, 0, 0 )


class PTF:
    def __init__(
            self,
            frameWidth: int,
            frameRatio: float,
            maxIter: int = 64,
            divergenceLimit: int = 1e+10
        ):
            self.frameWidth = frameWidth
            self.frameHeight = int(frameWidth * frameRatio)
            self.frameRatio = frameRatio
            self.maxIter = maxIter
            self.divergenceLimit = divergenceLimit

    def getImage(
        self,
        x: float,
        y: float,
        eps: float
    ) -> PTFImage:
        frame = np.full(
            shape = ( self.frameHeight, self.frameWidth, 3 ),
            fill_value = (0, 0, 0),
            dtype = np.uint8
        )

        plotFrame(
            frame,
            self.frameWidth,
            self.frameHeight,
            x,
            y,
            eps,
            self.maxIter,
            self.divergenceLimit
        )

        image = PTFImage(frame, self.frameWidth, self.frameHeight, x, y, eps)

        return image
    
    def saveStaticZoomVideo(
        self,
        fps: int,
        runningTime: float,
        x: float,
        y: float,
        startEps: float,
        endEps: float,
        easing: Callable[[float], float] = lambda t: t
    ):
        numberOfFrames = int(fps * runningTime)
        fileName = f"{os.getcwd()}/videos/ptf_{self.frameWidth}_{self.frameHeight}_{x}_{y}_{startEps}_{endEps}_{fps}fps.mp4"
        video = cv2.VideoWriter(
            filename = fileName,
            fourcc = cv2.VideoWriter_fourcc(*"MJPG"),
            fps = fps,
            frameSize = (self.frameWidth, self.frameHeight)
        )
        frame = np.full(
            shape = ( self.frameHeight, self.frameWidth, 3 ),
            fill_value = (0, 0, 0),
            dtype = np.uint8
        )

        epsStep = (endEps / startEps) ** (1 / numberOfFrames)

        for i in tqdm(range(numberOfFrames), leave = True):
            t = i / (numberOfFrames - 1)
            eps = startEps * (epsStep ** (easing(t) * numberOfFrames))
            plotFrame(
                frame,
                self.frameWidth,
                self.frameHeight,
                x,
                y,
                eps,
                self.maxIter,
                self.divergenceLimit
            )
            video.write(frame)
        
        video.release()
        print(f"Video saved successfully at {fileName}")
        cv2.destroyAllWindows()