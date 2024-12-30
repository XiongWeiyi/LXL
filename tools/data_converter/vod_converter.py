from pathlib import Path

import mmcv
import numpy as np

from mmdet3d.core.bbox import box_np_ops
from tools.data_converter.vod_data_utils import get_vod_image_info

vod_categories = ('Car', 'Pedestrian', 'Cyclist', 'rider', 'bicycle', 'bicycle_rack',
                  'human_depiction', 'moped_scooter', 'motor', 'truck',
                  'ride_other', 'vehicle_other', 'ride_uncertain')

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(infos, remove_outside=True):
    for info in mmcv.track_iter_progress(infos):
        lidar_info = info['lidar_point_cloud']
        radar_info = info['radar_point_cloud']
        image_info = info['image']
        calib = info['calib']

        lidar_path = lidar_info['velodyne_path']
        lidar_features = lidar_info['num_features']
        radar_path = radar_info['velodyne_path']
        radar_features = radar_info['num_features']
        points_l = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, lidar_features])  # [N_lidar, 4]
        points_r = np.fromfile(radar_path, dtype=np.float32, count=-1).reshape([-1, radar_features])  # [N_radar, 7]

        rect = calib['R0_rect']
        Trl2c = calib['Tr_lidar_to_cam']
        Trr2c = calib['Tr_radar_to_cam']
        Trr2l = np.linalg.inv(Trl2c) @ Trr2c
        P0 = calib['P0']
        if remove_outside:
            points_l = box_np_ops.remove_outside_points(points_l, rect, Trl2c, P0, image_info['image_shape'])
            points_r = box_np_ops.remove_outside_points(points_r, rect, Trr2c, P0, image_info['image_shape'])

        annos = info['annos']
        dims = annos['dimensions']
        loc = annos['location']
        rots = annos['rotation_y']
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)   # [num_obj, 3+3+1]
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trl2c)
        lidar_indices = box_np_ops.points_in_rbbox(points_l[:, :3], gt_boxes_lidar)    # [num_lidar_points, num_obj]
        points_r_in_l = Trr2l @ np.concatenate((points_r[:, :3], np.ones((len(points_r), 1))), axis=-1).T
        radar_indices = box_np_ops.points_in_rbbox(points_r_in_l.T[:, :3], gt_boxes_lidar)
        num_lidar_points_in_gt = lidar_indices.sum(0)       # [num_obj]
        num_radar_points_in_gt = radar_indices.sum(0)

        mask = annos['name'] == 'DontCare'
        num_lidar_points_in_gt[mask] = -1
        num_radar_points_in_gt[mask] = -1
        annos['num_lidar_points_in_gt'] = num_lidar_points_in_gt.astype(np.int32)
        annos['num_radar_points_in_gt'] = num_radar_points_in_gt.astype(np.int32)


def create_vod_info_file(data_path):               # .../view_of_delft_PUBLIC/
    """Create info file of View-of-Delft dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        data_path (str): Path of the data root.
    """
    imageset_folder = Path(data_path) / 'lidar/ImageSets'                # .../view_of_delft/lidar/ImageSets
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    save_path = Path(data_path)

    kitti_infos_train = get_vod_image_info(data_path, image_ids=train_img_ids)
    _calculate_num_points_in_gt(kitti_infos_train)
    filename = save_path / 'VoD_infos_train.pkl'                  # .../view_of_delft/vod_infos_train.pkl
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(kitti_infos_train, filename)

    kitti_infos_val = get_vod_image_info(data_path, image_ids=val_img_ids)
    _calculate_num_points_in_gt(kitti_infos_val)
    filename = save_path / 'VoD_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    mmcv.dump(kitti_infos_val, filename)

    filename = save_path / 'VoD_infos_trainval.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    mmcv.dump(kitti_infos_train + kitti_infos_val, filename)

    kitti_infos_test = get_vod_image_info(data_path, label_info=False, image_ids=test_img_ids)
    filename = save_path / 'VoD_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(kitti_infos_test, filename)


def _create_reduced_point_cloud(info_path):
    """Create reduced point clouds for given info.
    Args:
        info_path (str): Path of data info.
    """
    vod_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(vod_infos):
        lidar_info = info['lidar_point_cloud']
        radar_info = info['radar_point_cloud']
        image_info = info['image']
        calib = info['calib']

        lidar_path = Path(lidar_info['velodyne_path'])
        radar_path = Path(radar_info['velodyne_path'])
        lidar_features = lidar_info['num_features']
        radar_features = radar_info['num_features']
        points_lidar = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, lidar_features])
        points_radar = np.fromfile(str(radar_path), dtype=np.float32, count=-1).reshape([-1, radar_features])

        rect = calib['R0_rect']
        P0 = calib['P0']
        Trl2c = calib['Tr_lidar_to_cam']
        Trr2c = calib['Tr_radar_to_cam']

        points_lidar = box_np_ops.remove_outside_points(points_lidar, rect, Trl2c, P0, image_info['image_shape'])
        points_radar = box_np_ops.remove_outside_points(points_radar, rect, Trr2c, P0, image_info['image_shape'])
        save_dir_lidar = lidar_path.parent.parent / (lidar_path.parent.stem + '_reduced')   # .../view_of_delft_PUBLIC/lidar/training/velodyne_reduced
        save_dir_radar = radar_path.parent.parent / (radar_path.parent.stem + '_reduced')
        if not save_dir_lidar.exists():
            save_dir_lidar.mkdir()
        if not save_dir_radar.exists():
            save_dir_radar.mkdir()
        save_filename_lidar = save_dir_lidar / lidar_path.name      # .../view_of_delft_PUBLIC/lidar/training/velodyne_reduced/xxxxx.bin
        save_filename_radar = save_dir_radar / radar_path.name

        with open(save_filename_lidar, 'w') as f:
            points_lidar.tofile(f)
        with open(save_filename_radar, 'w') as f:
            points_radar.tofile(f)


def create_reduced_point_cloud(data_path):
    """Create reduced point clouds for training/validation/testing.
    Args:
        data_path (str): Path of original data.
    """
    train_info_path = Path(data_path) / 'VoD_infos_train.pkl'
    val_info_path = Path(data_path) / 'VoD_infos_val.pkl'
    test_info_path = Path(data_path) / 'VoD_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(train_info_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(val_info_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(test_info_path)