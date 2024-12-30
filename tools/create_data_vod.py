import argparse
from tools.data_converter import vod_converter as vod

def LXL_data_prep(root_path):            # .../view_of_delft_PUBLIC/
    """Prepare data related to View-of-Delft dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
    """
    vod.create_vod_info_file(root_path)
    vod.create_reduced_point_cloud(root_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--root-path', type=str, default='/mnt/e/view_of_delft_PUBLIC/',    # TODO: set path
                    help='specify the root path of dataset')
args = parser.parse_args()

if __name__ == '__main__':
    LXL_data_prep(root_path=args.root_path)