from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines import DefaultFormatBundle
from mmdet3d.core.points import BasePoints
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class DefaultFormatBundle3DV2(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self):
        super(DefaultFormatBundle3DV2, self).__init__()

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'lidar_points' in results:
            assert isinstance(results['lidar_points'], BasePoints)
            results['lidar_points'] = DC(results['lidar_points'].tensor)
        if 'radar_points' in results:
            assert isinstance(results['radar_points'], BasePoints)
            results['radar_points'] = DC(results['radar_points'].tensor)

        results = super(DefaultFormatBundle3DV2, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
