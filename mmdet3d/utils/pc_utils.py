import torch
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz:  N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    num_points=pred_xyz.shape[0]
    src_range=[
        torch.tensor(src_range[:3],device=pred_xyz.device),
        torch.tensor(src_range[3:6], device=pred_xyz.device)-1,
    ]
    if dst_range is None:
        dst_range = [
            torch.zeros(3, device=pred_xyz.device),
            torch.ones( 3, device=pred_xyz.device),
        ]

    # if pred_xyz.ndim == 4:
    #     src_range = [x[:, None] for x in src_range]
    #     dst_range = [x[:, None] for x in dst_range]

    # assert src_range[0].shape[0] == pred_xyz.shape[0]
    # assert dst_range[0].shape[0] == pred_xyz.shape[0]
    # assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    # assert src_range[0].shape == src_range[1].shape
    # assert dst_range[0].shape == dst_range[1].shape
    # assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][None,:] - src_range[0][None,:]
    dst_diff = dst_range[1][None,:] - dst_range[0][None,:]
    prop_xyz = (
        ((pred_xyz - src_range[0][None, :]) * dst_diff) / src_diff
    ) + dst_range[0][None, :]
    return prop_xyz

def shift_scale_points_2d(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz:  N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    num_points=pred_xyz.shape[0]
    src_range=[
        torch.tensor(src_range[:2],device=pred_xyz.device),
        torch.tensor(src_range[2:4], device=pred_xyz.device)-1,
    ]
    if dst_range is None:
        dst_range = [
            torch.zeros(2, device=pred_xyz.device),
            torch.ones( 2, device=pred_xyz.device),
        ]


    src_diff = src_range[1][None,:] - src_range[0][None,:]
    dst_diff = dst_range[1][None,:] - dst_range[0][None,:]
    prop_xyz = (
        ((pred_xyz - src_range[0][None, :]) * dst_diff) / src_diff
    ) + dst_range[0][None, :]
    return prop_xyz


def shift_scale_points_batch(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz