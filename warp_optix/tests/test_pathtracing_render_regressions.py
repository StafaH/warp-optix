# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import PathTracer

pytestmark = [pytest.mark.gpu, pytest.mark.render]


def _new_gpu_api(width=64, height=48):
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")

    api = PathTracer(
        width=width,
        height=height,
        denoiser="none",
        enable_set=False,
        backface_culling=False,
    )
    api.initialize()
    return api


def _render_glass_sheet(reverse_winding):
    api = _new_gpu_api()
    try:
        glass = api.scene.materials.add_glass(
            ior=1.45,
            tint=(0.58, 0.74, 1.0),
            transmission=0.97,
        )
        backdrop = api.create_emissive_material((0.8, 0.3, 0.05), intensity=3.0)
        quad = np.asarray(
            ((-1.4, -1.0, 0.0), (1.4, -1.0, 0.0), (1.4, 1.0, 0.0), (-1.4, 1.0, 0.0)),
            dtype=np.float32,
        )
        glass_indices = np.asarray(
            ((0, 2, 1), (0, 3, 2)) if reverse_winding else ((0, 1, 2), (0, 2, 3)),
            dtype=np.uint32,
        )
        backdrop_indices = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
        glass_geometry = api.create_mesh(
            quad + np.asarray((0.0, 0.0, 0.5), dtype=np.float32),
            glass_indices,
            material_id=glass,
        )
        backdrop_geometry = api.create_mesh(
            quad,
            backdrop_indices,
            material_id=backdrop,
        )
        api.create_instance(glass_geometry)
        api.create_instance(backdrop_geometry)
        api.set_camera_look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        api.build_scene()
        api.set_environment_color((0.02, 0.04, 0.08))

        frames = []
        for _ in range(4):
            api.render_frame()
            frames.append(api.get_frame()[..., :3].copy())
        return float(np.mean(frames))
    finally:
        api.close()


def test_two_sided_physical_glass_transmits_for_both_windings():
    """Catch double face orientation in the real thin-glass render path."""
    front_luminance = _render_glass_sheet(reverse_winding=False)
    back_luminance = _render_glass_sheet(reverse_winding=True)

    assert front_luminance > 0.2
    assert back_luminance > 0.2
    assert back_luminance / front_luminance == pytest.approx(1.0, rel=0.15)


def test_instance_updates_refit_rebuild_tlas_and_produce_motion_vectors():
    """Host instance updates refit the TLAS, periodically rebuild it, and drive motion vectors."""
    api = _new_gpu_api(width=96, height=64)
    try:
        material = api.create_emissive_material((1.0, 0.3, 0.05), intensity=2.0)
        quad = np.asarray(((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)), dtype=np.float32)
        geometry = api.create_mesh(quad, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32), material_id=material)
        instance = api.create_instance_with_transform(geometry, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        api.set_camera_look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        api.build_scene()
        scene = api.scene

        for _ in range(scene._max_tlas_updates):
            api.rebuild_tlas()
        assert scene._tlas_updates_since_build == scene._max_tlas_updates
        api.rebuild_tlas()
        assert scene._tlas_updates_since_build == 0

        api.render_frame()
        api.render_frame()
        assert float(np.max(np.abs(api._viewer._motion_buffer.numpy()))) < 5.0e-5
        api.set_instance_transform(instance, (0.25, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        api.rebuild_tlas()
        api.render_frame()
        magnitude = np.linalg.norm(api._viewer._motion_buffer.numpy(), axis=-1)
        assert float(np.max(magnitude)) > 1.0
        assert int(np.count_nonzero(magnitude > 1.0e-5)) > 100

        # Runtime instance additions must not upload into the old motion buffers.
        for count in (2, 3):
            api.create_instance(geometry)
            api.build_scene()
            api.render_frame()
            assert scene._device_instance_transforms.shape == (count, 12)
            assert api._viewer._prev_instance_transforms_buffer.shape == (count, 12)
            assert np.isfinite(api.get_frame()).all()
    finally:
        api.close()
