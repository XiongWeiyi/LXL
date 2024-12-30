from os import path as osp
import matplotlib.pyplot as plt
import mmcv
import numpy as np

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def draw_bev_bbox(ax, bboxes, color='blue', facecolor='none', linewidth=2, alpha=0.5, zorder=1):
    for bbox in bboxes:
        y = bbox[0]
        x = bbox[1]
        y_size = bbox[3]
        x_size = bbox[4]
        angle = -bbox[6] * 180 / np.pi
        x = x - (np.sqrt(x_size ** 2 + y_size ** 2) / 2) * np.sin(np.arctan2(x_size, y_size) - angle / 180 * np.pi)
        y = y - (np.sqrt(x_size ** 2 + y_size ** 2) / 2) * np.cos(np.arctan2(x_size, y_size) - angle / 180 * np.pi)

        rect = plt.Rectangle(xy=(x, y), width=x_size, height=y_size, angle=angle, edgecolor=color, facecolor=facecolor,
                             linewidth=linewidth, alpha=alpha, zorder=zorder)
        ax.add_patch(rect)
        ax.plot([bbox[1], bbox[1] + y_size / 2 * np.sin(-angle / 180 * np.pi)],
                [bbox[0], bbox[0] + y_size / 2 * np.cos(-angle / 180 * np.pi)],
                color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def visualize(pred_bboxes=None, gt_bboxes=None, img=None, sensor2img=None, points=None, bev_range=None,
              out_dir=None, file_name=None):
    if img is not None:
        result_path = osp.join(out_dir, file_name)
        mmcv.mkdir_or_exist(result_path)
        if gt_bboxes is not None:
            show_gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, origin=(0.5, 0.5, 0))
            img = draw_lidar_bbox3d_on_img(show_gt_bboxes, img, sensor2img, dict(), color=(61, 102, 255))
        if pred_bboxes is not None:
            show_pred_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))
            img = draw_lidar_bbox3d_on_img(show_pred_bboxes, img, sensor2img, dict(), color=(241, 101, 72))
        mmcv.imwrite(img, osp.join(result_path, f'{file_name}_img.png'))

    if points is not None:
        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(1, 1, 1)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(0, 0, markersize=10, color='red', marker='^')
        if gt_bboxes is not None:
            draw_bev_bbox(ax, gt_bboxes, color='orange', facecolor='gold', linewidth=2, alpha=1, zorder=1)
        plt.scatter(points[:, 1], points[:, 0], s=3, c='gray', zorder=2)
        if pred_bboxes is not None:
            draw_bev_bbox(ax, pred_bboxes, color='blue', facecolor='none', linewidth=2, alpha=1, zorder=3)
        ax.axis(bev_range)
        ax.invert_xaxis()
        ax.set_aspect(1)
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.1, wspace=0.1)
        plt.savefig(out_dir + file_name + '/' + file_name + "_bev.png")
        plt.close(figure)


