import torch

# from skimage import color
xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])

# only for speedup
lab_from_rgb = xyz_from_rgb / torch.Tensor([0.95047, 1.0, 1.08883]).view(3, 1)
t = lab_from_rgb[0].clone()
lab_from_rgb[0] = lab_from_rgb[1]; lab_from_rgb[1] = t


def rgb2xyz(rgb, dim=-1):
    assert isinstance(rgb, torch.Tensor)
    assert rgb.dtype == torch.uint8 or rgb.dtype == torch.float32
    assert rgb.shape[dim] == 3

    sh = rgb.shape
    if (rgb.dim() == 1):
        rgb = rgb.view(1, 3)
    
    rgb = rgb.transpose(dim, -1)

    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255

    xyz = torch.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = torch.matmul(xyz_from_rgb.to(rgb.device).view(1, 3, 3), xyz.view(-1, 3, 1))
    return xyz.view(rgb.shape).transpose(dim, -1).reshape(sh)

def xyz2lab(xyz, dim=-1):
    assert isinstance(xyz, torch.Tensor)
    assert xyz.dtype == torch.float32
    assert xyz.shape[dim] == 3

    sh = xyz.shape
    if (xyz.dim() == 1):
        xyz = xyz.view(1, 3)
    
    xyz = xyz.transpose(dim, -1)

    # Observer= 2°, Illuminant= D65
    xyz /= torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device)

    lab = torch.where(xyz > 0.008856, xyz ** (1/3), xyz * 7.787 + 16/116).view(-1, 3)
    # lab = torch.stack((116 * lab[:, 1] - 16, 500 * (lab[:, 0] - lab[:, 1]), 200 * (lab[:, 1] - lab[:, 2])), dim=1)

    t = lab[:, 0].clone()
    lab[:, 0] = lab[:, 1]
    lab[:, 1] = 500 * (t - lab[:, 0])
    lab[:, 2] = 200 * (lab[:, 0] - lab[:, 2])
    lab[:, 0] = lab[:, 0] * 116 - 16

    return lab.view(xyz.shape).transpose(dim, -1).reshape(sh)

def rgb2lab(rgb, dim=-1):
    assert isinstance(rgb, torch.Tensor)
    assert rgb.dtype == torch.uint8 or rgb.dtype == torch.float32
    assert rgb.shape[dim] == 3

    sh = rgb.shape
    if (rgb.dim() == 1):
        rgb = rgb.view(1, 3)
    
    rgb = rgb.transpose(dim, -1)

    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255

    res = torch.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    res = torch.matmul(lab_from_rgb.to(rgb.device).view(1, 3, 3), res.view(-1, 3, 1)).view(-1, 3)
    res = torch.where(res > 0.008856, res ** (1/3), res * 7.787 + 16/116)

    res[:, 1] = 500 * (res[:, 1] - res[:, 0])
    res[:, 2] = 200 * (res[:, 0] - res[:, 2])
    res[:, 0] = res[:, 0] * 116 - 16

    return res.view(rgb.shape).transpose(dim, -1).reshape(sh)
