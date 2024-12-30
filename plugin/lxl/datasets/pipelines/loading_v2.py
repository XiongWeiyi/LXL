import mmcv
import numpy as np

from mmdet3d.core.points import get_points_type
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromFileV2(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 sensor='lidar',    # 'lidar' or 'radar'
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 mean=None,
                 std=None,
                 file_client_args=dict(backend='disk')
                 ):
        assert sensor in ['lidar', 'radar']
        self.sensor = sensor
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        if mean is None:
            self.mean = np.zeros(load_dim, dtype=float)
        else:
            self.mean = np.array(mean, dtype=float)
        if std is None:
            self.std = np.ones(load_dim, dtype=float)
        else:
            self.std = np.array(std, dtype=float)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        sensor = self.sensor
        save_key = f'{sensor}_points'
        pts_filename = results[f'{sensor}_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)

        points = (points - self.mean) / self.std
        points = points[:, self.use_dim]

        attribute_dims = None

        points_class = get_points_type(self.coord_type)         # LiDARPoints
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results[save_key] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str
