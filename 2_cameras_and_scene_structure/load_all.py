from plots import *
from lib import *

def load_all():
    N = 12
    correspondences_dict = {}
        
    for i in range(1, N + 1):
        points_1 = np.loadtxt(f'scene/matches/u_{i:02d}.txt')
        for j in range(1, N + 1):
            if i >= j:
                continue
            points_2 = np.loadtxt(f'scene/matches/u_{j:02d}.txt')
            matches = np.loadtxt(f'scene/matches/m_{i:02d}_{j:02d}.txt').astype(int)
            correspondences = get_correspondences(matches, points_1, points_2)
            correspondences_dict[i, j] = correspondences
            correspondences_dict[j, i] = correspondences[[2,3,0,1]]
    
    return correspondences_dict