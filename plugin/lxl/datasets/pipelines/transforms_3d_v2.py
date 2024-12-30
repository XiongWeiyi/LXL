import numpy as np

from mmdet3d.datasets.pipelines import RandomFlip3D

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PointsRangeFilterV2(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range, sensor='lidar'):
        assert sensor in ['lidar', 'radar']
        self.sensor = sensor
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        sensor = self.sensor
        points = input_dict[f'{sensor}_points']

        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = input_dict[f'{sensor}_points'][points_mask]
        input_dict[f'{sensor}_points'] = clean_points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class PointShuffleV2(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        if 'lidar_points' in input_dict:
            input_dict['lidar_points'].shuffle()
        if 'radar_points' in input_dict:
            input_dict['radar_points'].shuffle()

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class RandomFlip3DV2(RandomFlip3D):
    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'radar_points' in input_dict:
                input_dict['radar_points'] = input_dict[key].flip(direction, points=input_dict['radar_points'])
            else:
                input_dict[key].flip(direction)