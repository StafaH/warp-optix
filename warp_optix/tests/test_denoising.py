# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import (
    DenoiserInputs,
    OptixDenoiser,
    PathTracer,
    Renderer,
)
from warp_optix.pathtracing import pathtracing_viewer as renderer_module


def _renderer(monkeypatch, mode):
    renderer = Renderer.__new__(Renderer)
    renderer.width, renderer.height = 64, 48
    renderer._render_width, renderer._render_height = 64, 48
    renderer.denoiser = mode
    renderer.optix_upscale = False
    renderer._denoiser_explicit = True
    renderer._device = object()
    renderer._render_stream = object()
    renderer._optix_denoiser = None
    renderer._sync_prev_camera_matrices_to_current = lambda: None
    renderer._set_render_resolution = lambda w, h: (
        setattr(renderer, "_render_width", w),
        setattr(renderer, "_render_height", h),
    )
    monkeypatch.setattr(renderer_module.wp, "synchronize_stream", lambda stream: None)
    return renderer


def test_auto_selects_optix_or_falls_back_but_explicit_backend_fails(monkeypatch):
    monkeypatch.setattr(renderer_module, "OptixDenoiser", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    renderer = _renderer(monkeypatch, "auto")
    renderer._init_denoiser()
    assert renderer.active_denoiser == "optix"
    assert renderer.denoiser_error is None

    def unavailable(*args, **kwargs):
        raise RuntimeError("OptiX unavailable")

    monkeypatch.setattr(renderer_module, "OptixDenoiser", unavailable)
    renderer._init_denoiser()
    assert renderer.active_denoiser == "none"
    assert "OptiX unavailable" in renderer.denoiser_error
    renderer.denoiser = "optix"
    with pytest.raises(RuntimeError, match="OptiX unavailable"):
        renderer._init_denoiser()


def test_failed_switch_restores_previous_backend(monkeypatch):
    renderer = _renderer(monkeypatch, "none")

    def create(*args, **kwargs):
        if not kwargs["upscale"]:
            raise RuntimeError("unavailable configuration")
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(renderer_module, "OptixDenoiser", create)
    renderer.set_denoiser("optix", optix_upscale=True)
    assert renderer.active_denoiser == "optix"
    assert (renderer._render_width, renderer._render_height) == (32, 24)
    with pytest.raises(RuntimeError, match="unavailable configuration"):
        renderer.set_denoiser("optix", optix_upscale=False)
    assert renderer.denoiser == renderer.active_denoiser == "optix"
    assert renderer.optix_upscale is True
    assert renderer._denoiser_reset_history is True
    assert renderer.sample_index == 0
    renderer.width = 63
    renderer._init_denoiser()
    assert renderer._render_width == 32
    # A closed renderer only records the choice; it creates no denoiser.
    renderer._device = None
    renderer.set_denoiser("optix", optix_upscale=False)


requires_gpu = pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")


def _cornell_api(**kwargs):
    api = PathTracer(enable_set=False, **kwargs)
    api.initialize()
    api.scene.create_cornell_box()
    api.build_scene()
    return api


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(("upscale", "other_context"), [(False, False), (True, True)])
def test_optix_gpu_apply_preserves_cuda_context(upscale, other_context):
    cuda = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]

    def current_context():
        context = ctypes.c_void_p()
        assert cuda.cuCtxGetCurrent(ctypes.byref(context)) == 0
        return context.value

    denoiser = OptixDenoiser(64, 48, upscale=upscale, device="cuda")
    inputs = DenoiserInputs(
        color=wp.ones((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        normal=wp.zeros((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        albedo=wp.ones((48, 64), dtype=wp.vec4, device=denoiser.stream.device),
        motion=wp.zeros((48, 64), dtype=wp.vec2, device=denoiser.stream.device),
    )
    original_context = current_context()
    caller_context = ctypes.c_void_p()
    try:
        # A second context on the same GPU exercises this without requiring two GPUs.
        if other_context:
            assert cuda.cuCtxCreate_v2(ctypes.byref(caller_context), 0, denoiser.stream.device.ordinal) == 0
        # Exercise fresh bindings, both cached output parities, and a history reset.
        for reset in (False, False, False, True):
            assert cuda.cuCtxSetCurrent(caller_context) == 0
            denoiser.apply(inputs, reset=reset)
            assert current_context() == caller_context.value
    finally:
        assert cuda.cuCtxSetCurrent(original_context) == 0
        denoiser.close()
        if caller_context.value is not None:
            assert cuda.cuCtxDestroy_v2(caller_context) == 0
        assert cuda.cuCtxSetCurrent(original_context) == 0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("upscale", [False, True])
def test_optix_gpu_denoises_and_resets_history(upscale):
    rng = np.random.default_rng(42)
    noisy = np.ones((48, 64, 4), dtype=np.float32)
    noisy[..., :3] += rng.normal(0.0, 0.15, (48, 64, 3)).astype(np.float32)
    normals = np.zeros_like(noisy)
    normals[..., 2] = 1.0
    inputs = DenoiserInputs(
        color=wp.array(noisy, dtype=wp.vec4, device="cuda"),
        normal=wp.array(normals, dtype=wp.vec4, device="cuda"),
        albedo=wp.ones((48, 64), dtype=wp.vec4, device="cuda"),
        motion=wp.zeros((48, 64), dtype=wp.vec2, device="cuda"),
    )
    denoiser = OptixDenoiser(64, 48, upscale=upscale, device="cuda")
    try:
        first = denoiser.apply(inputs).numpy()
        scale = 2 if upscale else 1
        assert first.shape == (48 * scale, 64 * scale, 4)
        assert np.isfinite(first).all()
        assert np.mean((first[..., :3] - 1.0) ** 2) < np.mean((noisy[..., :3] - 1.0) ** 2) * 0.5
        denoiser.apply(inputs).numpy()
        reset = denoiser.apply(inputs, reset=True).numpy()
        np.testing.assert_allclose(reset, first, rtol=1e-5, atol=1e-5)
    finally:
        denoiser.close()


@pytest.mark.gpu
@requires_gpu
def test_upscaling_denoiser_allocates_scratch_beyond_two_gib():
    if wp.get_device("cuda").free_memory < 6 * 2**30:
        pytest.skip("needs about 3 GiB of free device memory")
    denoiser = OptixDenoiser(1920, 1080, upscale=True, device="cuda")
    try:
        assert denoiser._scratch_size > 2**31
    finally:
        denoiser.close()


@pytest.mark.gpu
@pytest.mark.render
@requires_gpu
def test_renderer_gpu_optix_samples_motion_switch_and_resize():
    # OptiX launches cannot be graph-captured, so a requested graph stays inactive.
    api = _cornell_api(width=64, height=48, denoiser="optix", optix_upscale=True, enable_cuda_graphs=True)
    try:
        scene = api.scene
        frames = []
        for _ in range(3):
            api.render_frame()
            frames.append(api.viewer._color_buffer.numpy().copy())
        assert api.viewer.sample_index == 3
        assert not api.cuda_graph_active
        assert not np.array_equal(frames[-1], frames[-2]), "Ray samples froze"
        assert api.get_frame().shape == (48, 64, 4)

        previous = api.viewer.camera.position.copy()
        api.set_camera_look_at(previous + (1.0, 0.0, 0.0), api.viewer.camera.target)
        api.render_frame()
        assert not np.array_equal(api.viewer._color_buffer.numpy(), frames[-1]), "Camera changes did not reach OptiX"
        assert float(np.max(np.abs(api.viewer._motion_buffer.numpy()))) > 0.1, "Camera movement produced no motion"

        api.reset_temporal_history()
        api.set_denoiser("none")
        api.render_frame()
        assert api.active_denoiser == "none"
        api.set_denoiser("optix", optix_upscale=False)
        api.resize(80, 60)
        api.render_frame()
        assert api.scene is scene
        assert api.active_denoiser == "optix"
        assert api.get_frame().shape == (60, 80, 4)
        assert np.isfinite(api.get_frame()).all()
        api.set_denoiser("optix", optix_upscale=True)
        api.resize(81, 61)
        api.render_frame()
        assert api.get_frame().shape == (61, 81, 4)
    finally:
        api.close()


@pytest.mark.gpu
@requires_gpu
def test_optix_gpu_frame_is_ordered_when_called_from_another_stream():
    def first_frame(caller):
        api = PathTracer(
            width=384,
            height=256,
            denoiser="optix",
            enable_set=False,
            enable_cuda_graphs=False,
        )
        try:
            api.initialize()
            api.scene.create_cornell_box()
            api.build_scene()
            with wp.ScopedStream(caller):
                previous = wp.get_stream("cuda")
                api.render_frame()
                assert wp.get_stream("cuda") == previous
            wp.synchronize_device()
            return api.get_frame().copy()
        finally:
            api.close()

    expected = first_frame(None)
    actual = first_frame(wp.Stream("cuda"))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
