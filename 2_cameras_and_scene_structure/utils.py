import ge
import numpy as np

def save_points(X, cameras, name='out.ply'):
    # save points
    g = ge.ge.GePly(name)
    color1 = np.array([[255, 255, 255]], dtype=np.uint8).T
    color1 = np.repeat(color1, X.shape[1], axis=1)

    for idx, camera in cameras.items():
        C = -camera.P[:, 3]
        C[-1] *= -1
        X = np.c_[X, C]

    color2 = np.array([[255, 0, 0]], dtype=np.uint8).T
    color2 = np.repeat(color2, len(cameras.items()), axis=1)

    color = np.c_[color1, color2]
    g.points(X, color)
    g.close()
