import numpy as np
from ops.sky import Sky_Seg_Tool
from ops.utils import dpt2xyz
from .spiral import spiral_camera_poses, circular_camera_poses

class Trajectory_Generation():
    def __init__(self, 
                 scene = None,
                 method = 'spiral') -> None:
        '''
        method = 'spiral'/ rot' / 'spin'
        '''
        self._method = method
        self.forward_ratio  = scene.traj_forward_ratio
        self.backward_ratio = scene.traj_backward_ratio
        self.min_percentage = scene.traj_min_percentage
        self.max_percentage = scene.traj_max_percentage

    def _radius(self, xyz):
        # get range
        _min = np.percentile(xyz,self.min_percentage,axis=0)
        _max = np.percentile(xyz,self.max_percentage,axis=0)
        _range = _max - _min
        # set radius to mean range of three axes
        self.radius = np.mean(_range)

    def _traj_spiral(self, nframe):
        trajs = spiral_camera_poses(nframe, self.radius, self.forward_ratio, self.backward_ratio)
        # trajs = circular_camera_poses(nframe, self.radius)
        return trajs

    def __call__(self, nframe, xyz):
        if xyz.ndim > 2:
            xyz = xyz.reshape(-1,3)
        self._radius(xyz)
        if self._method == 'rot':
            trajs = self._traj_rot(nframe)
        elif self._method == 'spin':
            trajs = self._traj_spin(nframe)
        elif self._method == 'spiral':
            trajs = self._traj_spiral(nframe)
        else:
            raise TypeError('method = rot / spiral')
        return trajs

def _generate_trajectory(cfg, scene, nframes=None):
    method = scene.traj_type
    nframe = cfg.scene.traj.n_sample*6 if nframes is None else nframes
    sky,dpt,intrinsic = scene.frames[0].sky,scene.frames[0].dpt,scene.frames[0].intrinsic
    xyz = dpt2xyz(dpt,intrinsic)
    init_xyz = xyz[~sky]
    generator = Trajectory_Generation(scene=scene,method=method)
    traj = generator(nframe,init_xyz)
    return traj