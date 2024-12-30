import numpy as np
from vod.evaluation.kitti_official_evaluate import do_eval


def vod_eval(gt_annotations, dt_annotations, current_classes, pr_detail_dict=None, custom_method=0):
    """
        Same as get_official_eval_result() in vod.evaluation.kitti_official_evaluate (adding log info)
    """
    if custom_method == 0:
        print("Evaluating kitti by default")
    elif custom_method == 3:
        print("Evaluating kitti by ROI")

    # Original OpenPCDet code
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.50, 0.50, 0.7, 0.50, 0.5],  # image
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],  # bev
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])  # 3d
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)

    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'truck',
        4: 'bicycle',
        5: 'bicycle_rack',
        6: 'human_depiction',
        7: 'moped_scooter',
        8: 'motor',
        9: 'ride_other',
        10: 'ride_uncertain',
        11: 'rider',
        12: 'vehicle_other'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]

    if custom_method == 0:
        result_name = 'kitti'
    elif custom_method == 3:
        result_name = 'kitti_roi'
    result = ''

    # check whether alpha is valid
    compute_aos = True
    for anno in dt_annotations:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annotations, dt_annotations, current_classes, min_overlaps, compute_aos, pr_detail_dict=pr_detail_dict,
        custom_method=custom_method)

    result += '\n----------- AP Results ------------\n\n'
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_min_overlap, metric, class]
        # mAP result: [num_class, num_diff, num_min_overlap]
        for i in range(1, 2):  # min_overlaps.shape[0]
            result += ('{} 3D AP@{:.2f}: '.format(class_to_name[curcls], min_overlaps[i, 1, j]))

            if i == 1:
                result += '{:.4f}\n'.format(*mAP3d[j, :, i])
                ret_dict['%s_3d_all' % class_to_name[curcls]] = mAP3d[j, 0, 1]    # get j class, difficulty, second min_overlap

    # calculate mAP over all classes if there are multiple classes
    if len(current_classes) > 1:
        result += '\nOverall 3D AP: '
        if mAP3d is not None:
            mAP3d = mAP3d.mean(axis=0)
            result += '{:.4f}\n'.format(*mAP3d[:, 1])

    if custom_method == 0:
        return result, {'entire_area': ret_dict}
    elif custom_method == 3:
        return result, {'roi': ret_dict}
    else:
        raise NotImplementedError