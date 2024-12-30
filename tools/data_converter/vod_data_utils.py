from concurrent import futures as futures
from pathlib import Path

import numpy as np
from skimage import io


def get_image_index_str(img_idx):
    return '{:05d}'.format(img_idx)


def get_vod_info_path(idx,
                      prefix,
                      info_type='image_2',
                      file_tail='.png',
                      exist_check=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = Path(prefix)
    file_path = Path('training') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   exist_check=True):
    prefix = prefix + "lidar"
    return get_vod_info_path(idx, prefix, 'image_2', '.jpg', exist_check)


def get_label_path(idx,
                   prefix,
                   exist_check=True):
    prefix = prefix + "lidar"
    return get_vod_info_path(idx, prefix, 'label_2', '.txt', exist_check)


def get_velodyne_path(idx,
                      prefix,
                      sensor='lidar',        # "lidar","radar","radar_3frames" or "radar_5frames"
                      exist_check=True):
    assert sensor in ['lidar', 'radar', 'radar_3frames', 'radar_5frames']
    prefix = prefix + sensor
    return get_vod_info_path(idx, prefix, 'velodyne', '.bin', exist_check)



def get_calib_path(idx,
                   prefix,
                   sensor='lidar',          # 'lidar' or 'radar'
                   exist_check=True):
    assert sensor in ['lidar', 'radar']
    prefix = prefix + sensor
    return get_vod_info_path(idx, prefix, 'calib', '.txt', exist_check)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_vod_image_info(path,
                       radar_frames=5,
                       label_info=True,
                       image_ids=7481,
                       num_worker=8):
    """
    VoD annotation format:
    {
        image: {
            image_idx: ...
            image_path: ...
            image_shape: [2] array
        }
        lidar_point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        radar_point_cloud: {
            num_features: 7
            velodyne_path: ...
        }
        calib: {
            P0: [4, 4] array
            R0_rect: [4, 4] array
            Tr_lidar_to_cam: [4, 4] array
            Tr_radar_to_cam: [4, 4] array
        }
        annos: {
            name: [num_gt] ground truth name array;
                                    Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
            truncated: [num_gt] array
                                    Not used, only there to be compatible with KITTI format.
            occluded: [num_gt] array
                                    Integer (0,1,2) indicating occlusion state: 0 = fully visible,
                                        1 = partly occluded, 2 = largely occluded.
            alpha: [num_gt]         Observation angle of object, ranging [-pi..pi]
            bbox: [num_gt, 4]       2D bounding box of object in the image (0-based index):
                                        contains left, top, right, bottom pixel coordinates.
                                        This was automatically calculated from the 3D boxes.
            dimensions: [num_gt, 3] array;
                                    3D object dimensions: height, width, length (in meters)
            location: [num_gt, 3] array;
                                    3D object location x,y,z in camera coordinates (in meters)
            rotation_y: [num_gt] angle array;
                                    Rotation around -Z axis of the LiDAR sensor [-pi..pi]
            score: [num_gt] array;
                                    [1,1,1,...,1]
            index: [num_gt] array;
                                    [0,1,2,...,num_gt-1]
            group_ids: [num_gt] array;
                                    used for multi-part object;
        }
    }
    """
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}

        lidar_info = {'num_features': 4}
        radar_info = {'num_features': 7}

        lidar_info['velodyne_path'] = get_velodyne_path(idx, path, 'lidar')
        assert radar_frames in [1, 3, 5]
        if radar_frames == 1:
            radar_mid_path = 'radar'
        else:
            radar_mid_path = f'radar_{radar_frames}frames'
        radar_info['velodyne_path'] = get_velodyne_path(idx, path, radar_mid_path)

        image_info = {'image_idx': idx}
        image_info['image_path'] = get_image_path(idx, path)
        img_path = image_info['image_path']
        image_info['image_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)

        if label_info:
            label_path = get_label_path(idx, path)
            annotations = get_label_anno(label_path)
            info['annos'] = annotations

        info['image'] = image_info
        info['lidar_point_cloud'] = lidar_info
        info['radar_point_cloud'] = radar_info

        calib_info = {}
        lidar_calib_path = get_calib_path(idx, path, 'lidar')
        radar_calib_path = get_calib_path(idx, path, 'radar')
        with open(lidar_calib_path, 'r') as f:
            lidar_lines = f.readlines()
        with open(radar_calib_path, 'r') as f:
            radar_lines = f.readlines()

        P0 = np.array([float(info) for info in lidar_lines[0].split(' ')[1:13]]).reshape([3, 4])
        P0 = _extend_matrix(P0)

        R0_rect = np.array([float(info) for info in lidar_lines[4].split(' ')[1:10]]).reshape([3, 3])
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect

        Tr_lidar_to_cam = np.array([float(info) for info in lidar_lines[5].split(' ')[1:13]]).reshape([3, 4])
        Tr_radar_to_cam = np.array([float(info) for info in radar_lines[5].split(' ')[1:13]]).reshape([3, 4])
        Tr_lidar_to_cam = _extend_matrix(Tr_lidar_to_cam)
        Tr_radar_to_cam = _extend_matrix(Tr_radar_to_cam)

        calib_info['P0'] = P0
        calib_info['R0_rect'] = rect_4x4
        calib_info['Tr_lidar_to_cam'] = Tr_lidar_to_cam
        calib_info['Tr_radar_to_cam'] = Tr_radar_to_cam
        info['calib'] = calib_info

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)
