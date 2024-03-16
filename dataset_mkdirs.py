import os
from tqdm import tqdm
wsi_root_path = '/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/20220419_new_data/small_small_wsi_gnn_data'
wsi_paths = os.listdir(wsi_root_path)
match_dirs = ['00_NIML', '01_ASC-US', '02_LSIL', '03_ASC-H', '04_HSIL']
for wsi_path in tqdm(wsi_paths):
    ab_wsi_path = os.path.join(wsi_root_path, wsi_path)
    ab_wsi_path_train = os.path.join(ab_wsi_path, 'train')
    ab_wsi_path_val = os.path.join(ab_wsi_path, 'val')
    is_mkdir = [1, 1, 1, 1, 1]
    for dir in os.listdir(ab_wsi_path_train):
        for i, s in enumerate(match_dirs):
            if(s==dir):
                is_mkdir[i]=0
                break
    # 创建文件夹
    names = []
    for i, dir_mkdir in enumerate(is_mkdir):
        if(dir_mkdir==1):
            names.append(match_dirs[i])
    for name in names:
        ab_name = os.path.join(ab_wsi_path_train, name)
        if not os.path.exists(ab_name):
            os.mkdir(ab_name)
    for name in names:
        ab_name = os.path.join(ab_wsi_path_val, name)
        if not os.path.exists(ab_name):
            os.mkdir(ab_name)