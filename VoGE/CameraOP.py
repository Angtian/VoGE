from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix, CamerasBase, Transform3d


def get_projection_transform(camera, **kwargs):
    K = _get_sfm_calibration_matrix(
        camera._N,
        camera.device,
        kwargs.get("focal_length", camera.focal_length),
        kwargs.get("principal_point", camera.principal_point),
        orthographic=False,
    )

    transform = Transform3d(
        matrix=K.transpose(1, 2).contiguous(), device=camera.device
    )
    return transform