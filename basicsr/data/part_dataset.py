from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torch.nn import functional as F
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop,augment_fov, paired_random_crop_coords
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torch
import numpy as np
# from calculate_psf_kernel import get_patch_psf_batch
from basicsr.data.calculate_psf_kernel import get_patch_psf_batch
import numpy as np
import torch

def fill_tensor_with_patches_fast(input_tensor, output_size=(256, 256), step_size=32):
    """
    将输入张量的值填充到输出张量的每个 patch 中，输出形状为 (75, 256, 256)。

    参数:
        input_tensor: 输入张量，形状为 (64, 75)
        output_size: 输出张量的尺寸，默认为 (256, 256)
        step_size: patch 的步长，默认为 32

    返回:
        output_tensor: 填充后的张量，形状为 (75, 256, 256)
    """
    # 创建一个形状为 (75, 256, 256) 的零张量
    output_tensor = torch.zeros((75, output_size[0], output_size[1]), dtype=input_tensor.dtype)

    # 计算 patch 的数量
    num_patches_y = output_size[0] // step_size
    num_patches_x = output_size[1] // step_size
    total_patches = num_patches_y * num_patches_x

    # 确保输入张量的 patch 数量不超过总 patch 数
    assert input_tensor.size(0) <= total_patches, "输入张量的 patch 数量超过输出张量容量"

    # 生成所有 patch 的起始坐标
    y_indices = torch.arange(0, output_size[0], step_size)  # [0, 32, 64, ..., 224]
    x_indices = torch.arange(0, output_size[1], step_size)  # [0, 32, 64, ..., 224]
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')  # [8, 8], [8, 8]

    # 将坐标展平
    y_grid = y_grid.reshape(-1)  # [64]
    x_grid = x_grid.reshape(-1)  # [64]

    # 遍历所有 patch 并填充值
    for patch_idx in range(input_tensor.size(0)):
        # 提取当前 patch 的起始坐标
        y_start = y_grid[patch_idx]
        x_start = x_grid[patch_idx]

        # 提取输入张量的对应值 (75,)
        values = input_tensor[patch_idx]  # [75]

        # 将值广播到对应的 patch 区域
        output_tensor[:, y_start:y_start + step_size, x_start:x_start + step_size] = values[:, None, None]

    return output_tensor


import numpy as np

def generate_patch_coordinates(sub_top_left, sub_size=(256, 256), patch_size=32):
    """
    生成子图中所有32x32小块的左上角全局坐标，形状为 [64, 2]

    参数:
        sub_top_left: 子图的左上角坐标，格式为 (x_start, y_start)
        sub_size: 子图尺寸，格式为 (width, height)（默认(256, 256)）
        patch_size: 小块尺寸（默认32）

    返回:
        coordinates: 全局坐标数组，形状为 [64, 2]
    """
    x_start, y_start = sub_top_left
    num_patches_x = sub_size[0] // patch_size  # 每个方向上的小块数量
    num_patches_y = sub_size[1] // patch_size

    # 生成子图内部的相对坐标
    x_rel = np.arange(0, sub_size[0], patch_size)
    y_rel = np.arange(0, sub_size[1], patch_size)

    # 生成网格并展平
    x_grid, y_grid = np.meshgrid(x_rel, y_rel)
    coordinates_rel = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    # 转换为全局坐标
    coordinates_global = coordinates_rel + np.array([x_start, y_start])

    return coordinates_global.astype(int)  


@DATASET_REGISTRY.register()
class PART_Dataset(data.Dataset):
    """PART.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs and PSF.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PART_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        # self.txt_path = opt['txt_path'] if 'txt_path' in opt else None
        self.psf_path = opt['psf_path']
        self.psf = torch.from_numpy(np.load(self.psf_path))
        self.view_pos = np.linspace(0, 1, self.psf.shape[0])  # 视场位置
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        psf = self.psf

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, top_gt, left_gt = paired_random_crop_coords(img_gt, img_lq, gt_size, scale, gt_path)
            coords_batch = generate_patch_coordinates((top_gt, left_gt), sub_size=(gt_size,gt_size), patch_size=32)
            # print('coords_batch',coords_batch)
            patch_psf = get_patch_psf_batch(psf, self.view_pos,coords_batch,img_size=(1280,1920),patch_length=32,device='cpu')
            #[64,3,h,w] -> [64,3,5,5]
            patch_psf_resize = torch.nn.functional.interpolate(patch_psf, (5, 5), mode='bilinear', align_corners=False)
            #[64,3,5,5] -> [64,3*5*5]
            patch_psf_resize = patch_psf_resize.view(patch_psf_resize.size(0), -1)
            psf_map = fill_tensor_with_patches_fast(patch_psf_resize)
            #psf map 降采样
            psf_map = F.interpolate(psf_map.unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)

            #tensor [75,64,64] -> numpy [64,64,75] cpu
            psf_map = psf_map.squeeze(0).permute(1,2,0).cpu().numpy()
            # flip, rotation
            img_gt, img_lq, psf_map = augment_fov([img_gt, img_lq,psf_map], self.opt['use_hflip'], self.opt['use_rot'])

        else:
            coords_batch = generate_patch_coordinates((0, 0), sub_size=(1280,1920), patch_size=32)
            patch_psf = get_patch_psf_batch(psf, self.view_pos,coords_batch,img_size=(1280,1920),patch_length=32,device='cpu')
            #[2400,3,h,w] -> [2400,3,5,5]
            patch_psf_resize = torch.nn.functional.interpolate(patch_psf, (5, 5), mode='bilinear', align_corners=False)
            #[2400,3,5,5] -> [2400,3*5*5]
            patch_psf_resize = patch_psf_resize.view(patch_psf_resize.size(0), -1)
            #[75,1280,1920] -> [75,1280 /4,1920 /4]
            psf_map = fill_tensor_with_patches_fast(patch_psf_resize,output_size=(1280,1920))
            #psf map 降采样
            psf_map = F.interpolate(psf_map.unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)
            psf_map = psf_map.squeeze(0)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.opt['phase'] == 'train':
            psf_map = torch.from_numpy(psf_map.copy().transpose(2, 0, 1)).float()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape,'psf',psf_map.shape)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'psf': psf_map}

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class PART_test_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PART_test_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        self.txt_path = opt['txt_path'] if 'txt_path' in opt else None
        self.psf_path = opt['psf_path']
        self.view_pos = np.linspace(0, 1, 64)  # 视场位置
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        lens_name = self.gt_folder.split('/')[-2]
        print('lens_name',lens_name)
        psf_path = self.psf_path + '/' + lens_name + '.npy'
        psf = torch.from_numpy(np.load(psf_path)) #(64,3,h,w)
        # #rgb name
        # gt_name = gt_path.split('/')[-1].split('.')[0]

        # #read index txt file
        # txt_path = self.txt_path + '/' + gt_name + '.txt'
        # with open(txt_path, 'r', encoding='utf-8') as file:
        #     content = file.read().rstrip('\n')  # 读取整个文件内容

        # psf_path = self.psf_path + '/' + content[:-4] + '.npy'
        # psf = torch.from_numpy(np.load(psf_path)) #(64,3,h,w)
        # #(64,3,h,w) -> [1,3,h,w] 平均
        # psf = np.mean(psf, axis=0, keepdims=True)
        # #(1,3,h,w) -> （1,1,h,w） 平均
        # psf = np.mean(psf, axis=1, keepdims=True)
        # psf = psf.copy()
        # #（1,1,h,w）->(h,w,1)
        # psf = np.squeeze(psf)
        # psf = psf[..., None]
        # print('psf',psf.shape)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, top_gt, left_gt = paired_random_crop_coords(img_gt, img_lq, gt_size, scale, gt_path)
            coords_batch = generate_patch_coordinates((top_gt, left_gt), sub_size=(gt_size,gt_size), patch_size=32)
            # print('coords_batch',coords_batch)
            patch_psf = get_patch_psf_batch(psf, self.view_pos,coords_batch,img_size=(1280,1920),patch_length=32,device='cpu')
            #[64,3,h,w] -> [64,3,5,5]
            patch_psf_resize = torch.nn.functional.interpolate(patch_psf, (5, 5), mode='bilinear', align_corners=False)
            #[64,3,5,5] -> [64,3*5*5]
            patch_psf_resize = patch_psf_resize.view(patch_psf_resize.size(0), -1)
            psf_map = fill_tensor_with_patches_fast(patch_psf_resize)
            #psf map 降采样
            psf_map = F.interpolate(psf_map.unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)

            #tensor [75,64,64] -> numpy [64,64,75] cpu
            psf_map = psf_map.squeeze(0).permute(1,2,0).cpu().numpy()
            # flip, rotation
            img_gt, img_lq, psf_map = augment_fov([img_gt, img_lq,psf_map], self.opt['use_hflip'], self.opt['use_rot'])

        else:
            coords_batch = generate_patch_coordinates((0, 0), sub_size=(1280,1920), patch_size=32)
            patch_psf = get_patch_psf_batch(psf, self.view_pos,coords_batch,img_size=(1280,1920),patch_length=32,device='cpu')
            #[2400,3,h,w] -> [2400,3,5,5]
            patch_psf_resize = torch.nn.functional.interpolate(patch_psf, (5, 5), mode='bilinear', align_corners=False)
            #[2400,3,5,5] -> [2400,3*5*5]
            patch_psf_resize = patch_psf_resize.view(patch_psf_resize.size(0), -1)
            #[75,1280,1920] -> [75,1280 /4,1920 /4]
            psf_map = fill_tensor_with_patches_fast(patch_psf_resize,output_size=(1280,1920))
            #psf map 降采样
            psf_map = F.interpolate(psf_map.unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)
            psf_map = psf_map.squeeze(0)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.opt['phase'] == 'train':
            psf_map = torch.from_numpy(psf_map.copy().transpose(2, 0, 1)).float()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape,'psf',psf_map.shape)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'psf': psf_map}

    def __len__(self):
        return len(self.paths)
