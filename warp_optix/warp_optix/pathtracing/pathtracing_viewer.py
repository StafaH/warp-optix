# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OptiX Path Tracing Viewer.
Python/OptiX path tracing viewer with temporal denoising.

This viewer renders a scene using OptiX ray tracing with PBR materials,
displaying raw buffers (radiance, normals, depth, etc.) for debugging.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

import warp_optix as wo

from . import pathtracing_warp_kernels as pwk
from .camera import Camera, cross3
from .defaults import DEFAULT_VIEWER_HEIGHT, DEFAULT_VIEWER_WIDTH
from .denoising import DenoiserInputs, OptixDenoiser
from .environment_map import EnvironmentMap
from .lighting import RENDERER_RADIANCE_PER_NIT
from .scene import DYNAMIC_BUILD_FLAGS, STATIC_BUILD_FLAGS, Scene
from .sky_lut import generate_physical_sky_lut
from .tonemap import Tonemapper
from .volume_composite import VolumeParams, composite_volume

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CameraMatrices:
    """Camera matrices as flat float32 tuples for the launch parameters."""

    key: tuple
    view: tuple
    proj: tuple
    view_inv: tuple
    proj_inv: tuple
    view_proj: tuple
    position: tuple
    forward: tuple
    right: tuple
    up: tuple
    tan_half_fov: float


@wp.kernel
def _accumulate_sample(
    sample: wp.array2d(dtype=wp.vec4),
    accum: wp.array2d(dtype=wp.vec4),
    sample_index: int,
):
    y, x = wp.tid()
    s = sample[y, x]
    if sample_index == 0:
        accum[y, x] = s
    else:
        a = accum[y, x]
        accum[y, x] = a + (s - a) / float(sample_index + 1)


@wp.kernel(enable_backward=False)
def _prepare_device_camera(
    positions: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    up_directions: wp.array(dtype=wp.vec3),
    fovs: wp.array(dtype=wp.float32),
    camera_transform: wp.mat44,
    aspect: float,
    initialized: wp.array(dtype=wp.int32),
    states: wp.array(dtype=pwk.DeviceCameraState),
):
    position4 = camera_transform * wp.vec4(positions[0][0], positions[0][1], positions[0][2], 1.0)
    target4 = camera_transform * wp.vec4(targets[0][0], targets[0][1], targets[0][2], 1.0)
    up4 = camera_transform * wp.vec4(up_directions[0][0], up_directions[0][1], up_directions[0][2], 0.0)
    position = wp.vec3(position4[0], position4[1], position4[2])
    forward = wp.vec3(target4[0], target4[1], target4[2]) - position
    if wp.length(forward) <= 1.0e-8:
        forward = wp.vec3(0.0, 0.0, -1.0)
    else:
        forward = wp.normalize(forward)
    world_up = wp.vec3(up4[0], up4[1], up4[2])
    right = wp.cross(forward, world_up)
    if wp.length(right) <= 1.0e-8:
        right = wp.vec3(1.0, 0.0, 0.0)
    else:
        right = wp.normalize(right)
    up = wp.normalize(wp.cross(right, forward))
    tan_half_fov = wp.tan(wp.radians(wp.clamp(fovs[0], 5.0, 120.0)) * 0.5)

    state = states[0]
    if initialized[0] == 0:
        state.previous_position = position
        state.previous_forward = forward
        state.previous_right = right
        state.previous_up = up
        state.previous_tan_half_fov = tan_half_fov
        state.previous_aspect = aspect
        initialized[0] = 1
    else:
        state.previous_position = state.position
        state.previous_forward = state.forward
        state.previous_right = state.right
        state.previous_up = state.up
        state.previous_tan_half_fov = state.tan_half_fov
        state.previous_aspect = state.aspect
    state.position = position
    state.forward = forward
    state.right = right
    state.up = up
    state.tan_half_fov = tan_half_fov
    state.aspect = aspect
    states[0] = state


class Renderer:
    """
    OptiX path-tracing renderer.

    Renders a scene using hardware ray tracing with PBR materials.
    """

    # Output modes
    OUTPUT_FINAL = 0
    OUTPUT_RADIANCE = 1
    OUTPUT_DEPTH = 2
    OUTPUT_MOTION = 3
    OUTPUT_NORMAL = 4
    OUTPUT_ROUGHNESS = 5
    OUTPUT_DIFFUSE = 6
    OUTPUT_SPECULAR = 7
    OUTPUT_SPEC_HITDIST = 8

    def __init__(
        self,
        width: int = DEFAULT_VIEWER_WIDTH,
        height: int = DEFAULT_VIEWER_HEIGHT,
        scene_setup: Callable[[Scene], None] | None = None,
        camera: Camera | None = None,
        accumulate_samples: bool = False,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        russian_roulette_start_bounce: int = 3,
        use_halton_jitter: bool = True,
        enable_set: bool = True,
        enable_cuda_graphs: bool = True,
        enable_texture_mipmaps: bool = False,
        backface_culling: bool = True,
        denoiser: str | None = None,
        optix_upscale: bool = False,
    ):
        """
        Initialize the path tracing viewer.

        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
        self._render_width = width
        self._render_height = height
        self.frame_index = 0
        self.sample_index = 0
        self.accumulate_samples = accumulate_samples
        self.samples_per_frame = max(1, int(samples_per_frame))
        self.max_bounces = max(1, int(max_bounces))
        self._pipeline_max_bounces = self.max_bounces
        self.direct_light_samples = max(1, int(direct_light_samples))
        self.russian_roulette_start_bounce = max(1, int(russian_roulette_start_bounce))
        self.use_halton_jitter = bool(use_halton_jitter)
        self._denoiser_explicit = denoiser is not None
        self.denoiser = self._normalize_denoiser(denoiser if denoiser is not None else "auto")
        self.optix_upscale = bool(optix_upscale)
        self._optix_denoiser = None
        self.denoiser_error: str | None = None
        self.enable_set = bool(enable_set)
        self.enable_cuda_graphs = bool(enable_cuda_graphs)
        self._cuda_graph_error: str | None = None
        self.backface_culling = bool(backface_culling)
        self.enable_texture_mipmaps = bool(enable_texture_mipmaps)
        self._set_active = False
        self._render_stream = wp.get_stream("cuda")

        # Camera
        if camera is None:
            self.camera = Camera(
                position=(0.0, 0.0, 6.0),
                target=(0.0, 0.0, 0.0),
                fov=45.0,
                aspect_ratio=width / height,
            )
        else:
            self.camera = camera
            self.camera.set_aspect_ratio(width, height)

        # Optional external scene configuration callback
        self._scene_setup = scene_setup

        # Default to path-traced final output.
        self.output_mode = self.OUTPUT_FINAL

        # OptiX state (initialized in build())
        self._device = None
        self._pipeline = None

        # Scene
        self._scene = None

        # Tonemapper
        self._tonemapper = Tonemapper(width, height)

        # Output buffers
        self._color_buffer = None
        self._sky_lut_buffer = None
        self._sky_lut_state = None
        self._accum_buffer = None
        self._normal_roughness_buffer = None
        self._motion_buffer = None
        self._depth_buffer = None
        self._diffuse_buffer = None
        self._specular_buffer = None
        self._spec_hit_dist_buffer = None
        self._instance_transforms_buffer = None
        self._prev_instance_transforms_buffer = None
        self._prev_instance_transforms_valid = False

        self._launch_params = None
        self._param_values: dict = {}
        self._array_views: dict = {}
        self._sky_params = None
        self._sky_params_state = None
        self._instance_transform_count = 0
        self._device_camera_positions = None
        self._device_camera_targets = None
        self._device_camera_up = None
        self._device_camera_fovs = None
        self._device_camera_transform = wp.mat44(*np.eye(4, dtype=np.float32).reshape(-1))
        self._device_camera_initialized = None
        self._device_camera_state = None
        self._volume = None
        self._volume_params = None

        # CUDA surface objects
        self._color_surface = None
        self._denoiser_reset_history = True
        self._denoiser_status_reported = False
        self._last_jitter = (0.0, 0.0)

        self._camera_matrices_cache: _CameraMatrices | None = None
        # The previous frame's view-projection, for motion vectors, and the camera that accumulated.
        self._prev_mvp: tuple = ()
        self._accum_camera_key: tuple = ()
        self._sync_prev_camera_matrices_to_current()
        self._last_output_mode = self.output_mode

        # Physical sky defaults.
        self.sky_rgb_unit_conversion = (
            RENDERER_RADIANCE_PER_NIT,
            RENDERER_RADIANCE_PER_NIT,
            RENDERER_RADIANCE_PER_NIT,
        )
        self.sky_multiplier = 1.0
        self.sky_haze = 0.5
        self.sky_redblueshift = 0.05
        self.sky_saturation = 1.0
        self.sky_horizon_height = 0.0
        self.sky_ground_color = (0.4, 0.4, 0.4)
        self.sky_horizon_blur = 1.0
        self.sky_night_color = (0.0, 0.0, 0.0)
        self.sky_sun_disk_intensity = 1.0
        # Afternoon sky: 45-degree elevation,
        # 240-degree azimuth (west-southwest).
        self.sky_sun_direction = (-0.6123724, 0.7071068, -0.3535534)
        self.sky_sun_disk_scale = 1.0
        self.sky_sun_glow_intensity = 1.0
        self.sky_y_is_up = 1
        self.sky_grayscale = 0.5
        self.env_intensity = (1.0, 1.0, 1.0)
        self.analytic_light_intensity = 1.0
        self.emissive_material_intensity = 1.0
        self.env_rotation = 0.0
        self.use_path_regularization = True
        self.use_psr = True
        self.override_roughness = -1.0
        self.override_metallic = -1.0
        self.bitangent_flip = 1.0

        # Optional HDR environment map (lat-long, RGBA32F).
        self._env_map: EnvironmentMap | None = None

    @staticmethod
    def _normalize_denoiser(value: str) -> str:
        value = str(value).strip().lower()
        if value not in ("auto", "optix", "none"):
            raise ValueError("denoiser must be 'auto', 'optix', or 'none'")
        return value

    @property
    def active_denoiser(self) -> str:
        return "optix" if self._optix_denoiser is not None else "none"

    def set_denoiser(self, denoiser: str, *, optix_upscale: bool | None = None):
        """Switch reconstruction backends without rebuilding the scene.

        Explicit selections require the backend; 'auto' tries OptiX,
        then raw rendering. Backend-specific resolution settings are retained.
        """
        mode = self._normalize_denoiser(denoiser)
        upscale = self.optix_upscale if optix_upscale is None else bool(optix_upscale)
        if mode == self.denoiser and upscale == self.optix_upscale and self._denoiser_explicit:
            return
        previous = self.denoiser, self.optix_upscale, self._denoiser_explicit
        self.denoiser, self.optix_upscale, self._denoiser_explicit = mode, upscale, True
        if self._device is None:
            return
        wp.synchronize_stream(self._render_stream)
        try:
            self._init_denoiser()
        except Exception:
            self.denoiser, self.optix_upscale, self._denoiser_explicit = previous
            self._init_denoiser()
            raise

    def _destroy_optix_denoiser(self):
        if self._optix_denoiser is not None:
            self._optix_denoiser.close()
            self._optix_denoiser = None

    def _init_denoiser(self):
        # Release old reconstruction resources before replacing render targets
        # to avoid retaining two generations of allocations during large resizes.
        self._destroy_optix_denoiser()
        self.denoiser_error = None
        self._denoiser_status_reported = False
        self.sample_index = 0
        self.frame_index = 0
        self._denoiser_reset_history = True
        self._optix_launch_graph = None
        self._optix_graph_warmed = False
        self._prev_instance_transforms_valid = False
        self._sync_prev_camera_matrices_to_current()
        if self._device is None:
            self._set_render_resolution(self.width, self.height)
            return
        if self.denoiser in ("optix", "auto"):
            try:
                scale = 2 if self.optix_upscale else 1
                self._set_render_resolution(
                    (self.width + scale - 1) // scale,
                    (self.height + scale - 1) // scale,
                )
                self._optix_denoiser = OptixDenoiser(
                    self._render_width,
                    self._render_height,
                    upscale=self.optix_upscale,
                    stream=self._render_stream,
                    output_size=(self.width, self.height),
                )
            except Exception as exc:
                if self.denoiser == "optix":
                    raise
                self.denoiser_error = f"OptiX initialization failed: {exc}"
                logger.warning("Denoiser fallback: %s", self.denoiser_error)
                self._set_render_resolution(self.width, self.height)
        else:
            self._set_render_resolution(self.width, self.height)

    def _denoiser_inputs(self, color=None) -> DenoiserInputs:
        return DenoiserInputs(
            color=self._color_buffer if color is None else color,
            normal=self._normal_roughness_buffer,
            albedo=self._diffuse_buffer,
            motion=self._motion_buffer,
            depth=self._depth_buffer,
            specular_albedo=self._specular_buffer,
            specular_hit_distance=self._spec_hit_dist_buffer,
        )

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        russian_roulette_start_bounce: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust runtime ray budgets without rebuilding the OptiX pipeline."""
        if max_bounces is not None:
            value = int(max_bounces)
            if value < 1 or value > self._pipeline_max_bounces:
                raise ValueError(
                    "max_bounces must be in the range "
                    f"[1, {self._pipeline_max_bounces}]; construct the viewer with a "
                    "larger max_bounces value to raise the compiled limit"
                )
            self.max_bounces = value
        if direct_light_samples is not None:
            value = int(direct_light_samples)
            if value < 1:
                raise ValueError("direct_light_samples must be at least 1")
            self.direct_light_samples = value
        if russian_roulette_start_bounce is not None:
            value = int(russian_roulette_start_bounce)
            if value < 1:
                raise ValueError("russian_roulette_start_bounce must be at least 1")
            self.russian_roulette_start_bounce = value
        if samples_per_frame is not None:
            value = int(samples_per_frame)
            if value < 1:
                raise ValueError("samples_per_frame must be at least 1")
            self.samples_per_frame = value

    def set_backface_culling(self, enabled: bool) -> None:
        """Globally enable or disable triangle backface culling for every ray."""
        value = bool(enabled)
        if value == self.backface_culling:
            return
        self.backface_culling = value
        self.sample_index = 0
        self.frame_index = 0
        self._optix_launch_graph = None
        self._optix_graph_warmed = False
        self._denoiser_reset_history = True

    def set_volume(
        self,
        volume: wp.Volume | None,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        *,
        density_scale=1.0,
        step_size=0.01,
        cool_colors=(
            (0.0, 0.025, 0.18),
            (0.0, 0.20, 0.95),
            (0.0, 0.92, 0.82),
        ),
        warm_colors=(
            (0.30, 0.008, 0.0),
            (1.0, 0.075, 0.0),
            (1.0, 0.62, 0.0),
        ),
        emission=0.0,
        anisotropy=0.0,
        transfer_table=None,
        density_feature=False,
    ):
        """Set the NanoVDB volume composited by the path tracer."""
        if transfer_table is None:
            table = None
        elif isinstance(transfer_table, wp.array):
            if transfer_table.ndim != 1 or transfer_table.dtype != wp.vec4 or len(transfer_table) < 2:
                raise ValueError("transfer_table must be a 1D Warp vec4 array with at least two entries")
            if not transfer_table.device.is_cuda:
                raise ValueError("transfer_table must be CUDA-resident")
            table = transfer_table
        else:
            table = np.asarray(transfer_table, dtype=np.float32)
            if table.ndim != 2 or table.shape[1] != 4 or len(table) < 2:
                raise ValueError("transfer_table must have shape (N, 4) with N >= 2")
            if not np.all(np.isfinite(table)) or np.any(table < 0.0):
                raise ValueError("transfer_table entries must be finite and nonnegative")
            table = wp.array(table, dtype=wp.vec4, device="cuda")
        # VolumeParams holds only the volume's id, so keep the volume itself alive.
        self._volume = volume
        self._volume_params = None
        if volume is not None:
            p = VolumeParams()
            p.volume = wp.uint64(volume.id)
            p.bounds_min = wp.vec3(*bounds_min)
            p.bounds_max = wp.vec3(*bounds_max)
            p.density_scale = float(density_scale)
            p.step_size = float(step_size)
            p.cool_dark, p.cool_mid, p.cool_light = (wp.vec3(*color) for color in cool_colors)
            p.warm_dark, p.warm_mid, p.warm_light = (wp.vec3(*color) for color in warm_colors)
            p.emission = float(emission)
            p.anisotropy = float(anisotropy)
            p.transfer_table = table
            p.transfer_count = wp.uint32(0 if table is None else len(table))
            p.density_feature = wp.uint32(bool(density_feature))
            self._volume_params = p
        self.sample_index = 0
        self.frame_index = 0
        self._denoiser_reset_history = True

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        """
        Load an HDR environment map from disk.

        The environment map is used for image-based lighting with importance sampling.

        Args:
            hdr_path: Path to HDR file (.hdr format)
            scaling: Intensity multiplier (default 1.0)
        """
        env_map = EnvironmentMap()
        if env_map.load_from_file(hdr_path, scaling=scaling):
            self._env_map = env_map
        else:
            logger.warning("Failed to load HDR environment: %s", hdr_path)

    def set_environment_color(self, color: tuple[float, float, float]):
        """
        Set a uniform color environment (useful for debugging or simple scenes).

        Args:
            color: RGB color values
        """
        env_map = EnvironmentMap()
        if env_map.load_from_color(color):
            self._env_map = env_map

    def clear_environment_map(self):
        """Clear the HDR environment map and use procedural sky only."""
        self._env_map = None

    @property
    def tonemapped_output(self):
        """Return the tonemapped output buffer used for display/extraction."""
        return self._tonemapper.output

    @property
    def linear_depth_output(self):
        """Return the current positive view-space depth buffer."""
        return self._depth_buffer

    @property
    def render_resolution(self) -> tuple[int, int]:
        """Return the internal render resolution as ``(width, height)``."""
        return self._render_width, self._render_height

    def build(self):
        """Build the OptiX pipeline and scene."""
        logger.info("Initializing OptiX path tracing viewer.")

        if self._device is not None:
            self.close()
        self._device = wo.get_context("cuda").device
        try:
            self._set_active = False

            # Create scene
            self._scene = Scene(enable_texture_mipmaps=self.enable_texture_mipmaps, device=self._device)
            if self._scene_setup is not None:
                self._scene_setup(self._scene)
            else:
                self._scene.create_cornell_box()
            self._scene.build()

            # Create output buffers
            self._create_buffers()
            self._reset_launch_params()
            self._init_denoiser()

            self._create_pipeline()

            logger.info("OptiX path tracing viewer build complete.")
        except BaseException:
            self.close()
            raise

    def _set_render_resolution(self, render_width: int, render_height: int):
        rw = max(1, int(render_width))
        rh = max(1, int(render_height))
        if rw == self._render_width and rh == self._render_height:
            return
        self._render_width = rw
        self._render_height = rh
        self._create_buffers()
        self.frame_index = 0
        self.sample_index = 0
        self._denoiser_reset_history = True

    def _sync_prev_camera_matrices_to_current(self):
        """Initialize previous-frame camera transforms from the current camera pose.

        Mirrors reference first-frame behavior where prevMVP is set to currentMVP to avoid
        spurious large motion vectors after resets/resizes.
        """
        camera = self._camera_matrices()
        self._prev_mvp = camera.view_proj
        self._accum_camera_key = camera.key

    def _camera_matrices(self) -> _CameraMatrices:
        """The camera's matrices, recomputed only when the camera changes."""
        c = self.camera
        key = (tuple(c.position), tuple(c.target), tuple(c.up), c.fov, c.aspect_ratio, c.near, c.far)
        cached = self._camera_matrices_cache
        if cached is None or cached.key != key:
            view, proj = c.get_view_matrix(), c.get_projection_matrix()

            def flat(m):
                return tuple(np.asarray(m, dtype=np.float32).reshape(-1).tolist())

            cached = self._camera_matrices_cache = _CameraMatrices(
                key,
                flat(view),
                flat(proj),
                flat(np.linalg.inv(view)),
                flat(np.linalg.inv(proj)),
                flat(view @ proj),
                *(flat(v) for v in self._compute_camera_basis()),
                float(np.tan(np.deg2rad(c.fov * 0.5))),
            )
        return cached

    def _create_buffers(self):
        """Create output buffers."""
        # HDR color buffer
        self._color_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._sky_lut_buffer = wp.empty((1024, 2048), dtype=wp.vec4, device="cuda")
        self._sky_lut_state = None
        self._accum_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")

        # G-buffer outputs
        self._normal_roughness_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.vec4, device="cuda"
        )
        self._motion_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec2, device="cuda")
        self._depth_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.float32, device="cuda")
        self._diffuse_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._specular_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._spec_hit_dist_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.float32, device="cuda"
        )

    def _update_instance_transform_buffers(self):
        """Upload current instance transforms for motion vectors.

        The *previous* transforms buffer is populated by
        :meth:`_snapshot_instance_transforms` which copies the current
        buffer to the previous buffer **after** each rendered frame.
        This guarantees that ``prev`` always holds exactly the transforms
        that were used for the last rendered frame.
        """
        if self._scene is None or self._scene.instance_count == 0:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        count = self._scene.instance_count

        self._instance_transforms_buffer = self._scene._device_instance_transforms
        if self._instance_transforms_buffer is None:
            raise RuntimeError("Scene instance transform buffers have not been built")

        # (Re)allocate GPU buffers when the instance count changes.
        if count != self._instance_transform_count:
            self._prev_instance_transforms_buffer = wp.empty((count, 12), dtype=wp.float32, device="cuda")
            self._instance_transform_count = count
            self._prev_instance_transforms_valid = False

        # First frame (or after resize): prev == current so motion is zero.
        if not self._prev_instance_transforms_valid:
            wp.copy(
                self._prev_instance_transforms_buffer,
                self._instance_transforms_buffer,
                stream=self._render_stream,
            )
            self._prev_instance_transforms_valid = True

    def _snapshot_instance_transforms(self):
        """Copy current instance transforms to the previous-frame buffer on GPU.

        Must be called once per frame **after** the OptiX launch so that the
        next frame sees the correct previous-frame transforms for rigid-body
        motion vectors.
        """
        if (
            self._instance_transforms_buffer is not None
            and self._prev_instance_transforms_buffer is not None
            and self._instance_transforms_buffer.shape == self._prev_instance_transforms_buffer.shape
        ):
            wp.copy(
                self._prev_instance_transforms_buffer,
                self._instance_transforms_buffer,
                stream=self._render_stream,
            )

    def _create_pipeline(self):
        """Create the OptiX pipeline with primary and shadow/visibility ray types.

        Geometry SBT offsets select the hit groups: triangles 0, then round linear
        and round cubic Bezier curves at 2 and 4 when static and 6 and 8 when
        dynamic, since built-in curve intersection depends on the build flags.
        Bounces iterate in raygen, so hit and miss programs never trace recursively.
        """
        curve_groups = []
        for flags in (STATIC_BUILD_FLAGS, DYNAMIC_BUILD_FLAGS):
            for builtin in (wo.PrimitiveType.ROUND_LINEAR, wo.PrimitiveType.ROUND_CUBIC_BEZIER):
                curve_groups += [
                    wo.HitGroup(pwk.curve_primary_closest_hit, pwk.primary_any_hit, builtin=builtin, build_flags=flags),
                    wo.HitGroup(pwk.secondary_closest_hit, pwk.secondary_any_hit, builtin=builtin, build_flags=flags),
                ]
        self._pipeline = wo.Pipeline(
            pwk.primary_raygen,
            miss=[pwk.primary_miss, pwk.secondary_miss],
            hit_groups=[
                wo.HitGroup(pwk.primary_closest_hit, pwk.primary_any_hit),
                wo.HitGroup(pwk.secondary_closest_hit, pwk.secondary_any_hit),
                *curve_groups,
            ],
            ray_types=2,
            payload_words=19,
            graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
            device=self._device,
        )

    @staticmethod
    def _halton(index: int, base: int) -> float:
        f = 1.0
        r = 0.0
        i = max(0, int(index))
        b = max(2, int(base))
        while i > 0:
            f /= float(b)
            r += f * float(i % b)
            i //= b
        return r

    def _compute_camera_basis(self):
        pos = np.asarray(self.camera.position, dtype=np.float32)
        target = np.asarray(self.camera.target, dtype=np.float32)
        world_up = np.asarray(self.camera.up, dtype=np.float32)

        forward = target - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1.0e-8:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= forward_norm

        right = cross3(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm

        up = cross3(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm < 1.0e-8:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            up /= up_norm

        return pos, forward, right, up

    def bind_device_camera(
        self,
        positions: wp.array,
        targets: wp.array,
        *,
        fov: float | wp.array = 45.0,
        up=(0.0, 1.0, 0.0),
        camera_transform: np.ndarray | None = None,
    ) -> None:
        """Bind graph-written CUDA eye and target arrays to the renderer."""
        if positions.dtype != wp.vec3 or targets.dtype != wp.vec3:
            raise TypeError("positions and targets must be wp.array[wp.vec3]")
        if positions.shape[0] < 1 or targets.shape[0] < 1:
            raise ValueError("positions and targets must contain at least one element")
        if positions.device != targets.device or not positions.device.is_cuda:
            raise ValueError("positions and targets must share a CUDA device")
        self._device_camera_positions = positions
        self._device_camera_targets = targets
        self._device_camera_up = wp.array([up], dtype=wp.vec3, device=positions.device)
        if hasattr(fov, "dtype"):
            if fov.dtype != wp.float32 or fov.shape[0] < 1 or fov.device != positions.device:
                raise ValueError("fov must be a non-empty wp.array[float] on the camera device")
            self._device_camera_fovs = fov
        else:
            self._device_camera_fovs = wp.array([float(fov)], dtype=float, device=positions.device)
        transform = np.eye(4, dtype=np.float32) if camera_transform is None else camera_transform
        transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        self._device_camera_transform = wp.mat44(*transform.reshape(-1))
        self._device_camera_initialized = wp.zeros(1, dtype=wp.int32, device=positions.device)
        self._device_camera_state = wp.empty(1, dtype=pwk.DeviceCameraState, device=positions.device)

    def _update_device_camera(self) -> None:
        if self._device_camera_state is None:
            return
        wp.launch(
            _prepare_device_camera,
            dim=1,
            inputs=[
                self._device_camera_positions,
                self._device_camera_targets,
                self._device_camera_up,
                self._device_camera_fovs,
                self._device_camera_transform,
                float(self._render_width) / float(max(1, self._render_height)),
                self._device_camera_initialized,
            ],
            outputs=[self._device_camera_state],
            device=self._device_camera_positions.device,
            stream=self._render_stream,
        )

    def unbind_device_camera(self) -> None:
        """Restore the host-driven camera path."""
        self._device_camera_positions = None
        self._device_camera_targets = None
        self._device_camera_up = None
        self._device_camera_fovs = None
        self._device_camera_initialized = None
        self._device_camera_state = None

    def _update_launch_params(
        self,
        frame_index_override: int | None = None,
        *,
        update_instance_transforms: bool = True,
        camera: _CameraMatrices | None = None,
    ):
        """Update launch parameters for the current frame."""
        if update_instance_transforms:
            self._update_instance_transform_buffers()
        if camera is None:
            camera = self._camera_matrices()

        if self._launch_params is None:
            self._reset_launch_params()

        p = self._launch_params
        frame_index = self.sample_index if frame_index_override is None else int(frame_index_override)
        if self.use_halton_jitter and self._optix_denoiser is None:
            self._last_jitter = (self._halton(frame_index, 2) - 0.5, self._halton(frame_index, 3) - 0.5)
        else:
            self._last_jitter = (0.0, 0.0)
        env_map = self._env_map
        flags = (
            (1 if env_map is None else 0)
            | (2 if self.use_psr else 0)
            | (4 if self.use_path_regularization else 0)
            | (8 if self.backface_culling else 0)
        )
        values = {
            "tlas": self._scene.tlas_handle,
            "width": self._render_width,
            "height": self._render_height,
            "frame_index": frame_index,
            "max_bounces": self.max_bounces,
            "direct_light_samples": self.direct_light_samples,
            "russian_roulette_start_bounce": self.russian_roulette_start_bounce,
            "analytic_light_intensity": float(self.analytic_light_intensity),
            "emissive_material_intensity": float(self.emissive_material_intensity),
            "output_mode": int(self.OUTPUT_FINAL),
            "cam_pos": camera.position,
            "cam_forward": camera.forward,
            "cam_right": camera.right,
            "cam_up": camera.up,
            "cam_tan_half_fov": camera.tan_half_fov,
            "cam_aspect": self._render_width / max(1, self._render_height),
            "jitter": self._last_jitter,
            "view": camera.view,
            "proj": camera.proj,
            "view_inv": camera.view_inv,
            "proj_inv": camera.proj_inv,
            "prev_mvp": self._prev_mvp,
            "env_intensity": tuple(float(v) for v in self.env_intensity),
            "ambient_light": tuple(float(v) for v in getattr(self._scene, "usd_ambient_light", (0.0, 0.0, 0.0))),
            "env_rotation": float(self.env_rotation),
            "flags": flags,
            "override_roughness": float(self.override_roughness),
            "override_metallic": float(self.override_metallic),
            "bitangent_flip": float(self.bitangent_flip),
            "use_procedural_sky": 1 if env_map is None else 0,
            "env_map_length": 0 if env_map is None else env_map.width * env_map.height * 4,
            "env_map_width": 0 if env_map is None else env_map.width,
            "env_map_height": 0 if env_map is None else env_map.height,
            "env_accel_count": 0 if env_map is None else env_map.accel_count,
            "sky_lut_width": self._sky_lut_buffer.shape[1],
            "sky_lut_height": self._sky_lut_buffer.shape[0],
            "sphere_light_count": self._scene.light_count,
            "render_prim_count": self._scene.geometry_count,
            "instance_count": self._scene.instance_count,
            "material_count": self._scene.materials.count,
            "texture_count": self._scene.texture_count,
        }
        # Most fields keep their value across frames, and assigning a struct field converts it.
        last = self._param_values
        for name, value in values.items():
            if last.get(name) != value:
                setattr(p, name, value)
                last[name] = value

        sd = self.sky_sun_direction
        sd_len = (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) ** 0.5
        if sd_len > 1.0e-8:
            sd = (sd[0] / sd_len, sd[1] / sd_len, sd[2] / sd_len)
        # The lookup table depends only on these; the sun disk, glow, and grayscale apply at shading.
        sky_lut_state = (
            tuple(self.sky_rgb_unit_conversion),
            float(self.sky_multiplier),
            float(self.sky_haze),
            float(self.sky_redblueshift),
            float(self.sky_saturation),
            float(self.sky_horizon_height),
            tuple(self.sky_ground_color),
            float(self.sky_horizon_blur),
            tuple(self.sky_night_color),
            tuple(sd),
            int(self.sky_y_is_up),
        )
        sky_state = (
            *sky_lut_state,
            float(self.sky_sun_disk_intensity),
            float(self.sky_sun_disk_scale),
            float(self.sky_sun_glow_intensity),
            float(self.sky_grayscale),
        )
        if sky_state != self._sky_params_state:
            sky = pwk.PhysicalSkyParams()
            sky.rgb_unit_conversion = wp.vec3(*self.sky_rgb_unit_conversion)
            sky.multiplier = float(self.sky_multiplier)
            sky.haze = float(self.sky_haze)
            sky.redblueshift = float(self.sky_redblueshift)
            sky.saturation = float(self.sky_saturation)
            sky.horizon_height = float(self.sky_horizon_height)
            sky.ground_color = wp.vec3(*self.sky_ground_color)
            sky.horizon_blur = float(self.sky_horizon_blur)
            sky.night_color = wp.vec3(*self.sky_night_color)
            sky.sun_disk_intensity = float(self.sky_sun_disk_intensity)
            sky.sun_direction = wp.vec3(*sd)
            sky.sun_disk_scale = float(self.sky_sun_disk_scale)
            sky.sun_glow_intensity = float(self.sky_sun_glow_intensity)
            sky.y_is_up = int(self.sky_y_is_up)
            sky.grayscale = float(self.sky_grayscale)
            p.sky = self._sky_params = sky
            self._sky_params_state = sky_state
        if env_map is None and sky_lut_state != self._sky_lut_state:
            wp.launch(
                generate_physical_sky_lut,
                dim=self._sky_lut_buffer.shape,
                inputs=[self._sky_params],
                outputs=[self._sky_lut_buffer],
                device="cuda",
                stream=self._render_stream,
            )
            self._sky_lut_state = sky_lut_state

        scene = self._scene
        arrays = {
            "device_camera": self._device_camera_state,
            "env_map": None if env_map is None else env_map._env_map_buffer,
            "env_accel": self._array_view(
                "env_accel",
                None if env_map is None else env_map._env_accel_buffer,
                0 if env_map is None else env_map.accel_count,
                pwk.EnvAccel,
            ),
            "sky_lut": self._sky_lut_buffer,
            "sphere_lights": self._array_view(
                "sphere_lights", scene._sphere_light_data, scene.light_count, pwk.SphereLight
            ),
            "render_primitives": self._array_view(
                "render_primitives", scene._render_primitives, scene.geometry_count, pwk.RenderPrimitive
            ),
            "instance_render_prim_ids": scene._instance_render_prim_ids,
            "instance_geometry_types": scene._instance_geometry_types,
            "instance_material_ids": scene._instance_material_ids,
            "instance_transforms": self._array_view(
                "instance_transforms", self._instance_transforms_buffer, scene.instance_count, pwk.TransformMatrix3x4
            ),
            "prev_instance_transforms": self._array_view(
                "prev_instance_transforms",
                self._prev_instance_transforms_buffer,
                scene.instance_count,
                pwk.TransformMatrix3x4,
            ),
            "compact_materials": self._array_view(
                "compact_materials", scene._compact_materials, scene.materials.count, pwk.CompactMaterial
            ),
            "packed_indices": scene._packed_indices,
            "packed_positions": scene._packed_positions,
            "packed_radii": scene._packed_radii,
            "packed_normals": scene._packed_normals,
            "packed_tangents": scene._packed_tangents,
            "packed_texcoords0": scene._packed_texcoords0,
            "packed_texcoords1": scene._packed_texcoords1,
            "packed_prev_positions": scene._packed_prev_positions,
            "packed_material_ids": scene._packed_material_ids,
            "textures": scene._texture_data,
            "color_output": self._color_buffer,
            "normal_roughness_output": self._normal_roughness_buffer,
            "motion_output": self._motion_buffer,
            "depth_output": self._depth_buffer,
            "diffuse_output": self._diffuse_buffer,
            "specular_output": self._specular_buffer,
            "spec_hit_dist_output": self._spec_hit_dist_buffer,
        }
        # Assigning an array field re-validates it, and the arrays change only with the scene or size.
        for name, value in arrays.items():
            if getattr(p, name) is not value:
                setattr(p, name, value)

    def _reset_launch_params(self) -> None:
        """Start a new launch-parameter struct; every field is assigned on the next update."""
        self._launch_params = pwk.PathtraceLaunchParams()
        self._param_values = {}
        self._sky_params_state = None

    def _array_view(self, name: str, buffer, count: int, dtype):
        """A typed view of a scene byte buffer for field ``name``, reused while the buffer and count are unchanged."""
        if buffer is None or count == 0:
            return None
        cached = self._array_views.get(name)
        if cached is None or cached.ptr != buffer.ptr or cached.shape[0] != count:
            cached = self._array_views[name] = wp.array(ptr=buffer.ptr, shape=(count,), dtype=dtype)
        return cached

    def _update_temporal_state(self, camera: _CameraMatrices, use_external_accum: bool) -> bool:
        """Update accumulation state and return whether temporal history resets."""
        if self._optix_denoiser is not None:
            return False

        reset_temporal = self.output_mode != self._last_output_mode or camera.key != self._accum_camera_key

        if use_external_accum and reset_temporal:
            self.frame_index = 0
        return bool(reset_temporal)

    def _launch_samples(
        self,
        samples_this_frame: int,
        use_external_accum: bool,
        *,
        camera: _CameraMatrices,
    ):
        """Launch the frame's samples and return the buffer holding the frame."""
        accumulate = use_external_accum or samples_this_frame > 1
        for s in range(samples_this_frame):
            launch_frame_index = self.sample_index + s
            self._update_launch_params(
                frame_index_override=launch_frame_index,
                update_instance_transforms=False,
                camera=camera,
            )

            self._launch_optix()

            if self._volume_params is not None:
                wp.launch(
                    composite_volume,
                    dim=(self._render_height, self._render_width),
                    inputs=[self._launch_params, self._volume_params],
                    device="cuda",
                    stream=self._render_stream,
                )

            if accumulate:
                accum_sample_index = int(self.frame_index if use_external_accum else s)
                wp.launch(
                    _accumulate_sample,
                    dim=(self._render_height, self._render_width),
                    inputs=[self._color_buffer, self._accum_buffer, accum_sample_index],
                    device="cuda",
                    stream=self._render_stream,
                )

            if use_external_accum:
                self.frame_index += 1
        return self._accum_buffer if accumulate else self._color_buffer

    def _launch_optix(self):
        if self.enable_cuda_graphs and self._cuda_graph_error is None:
            self._cuda_graph_error = "OptiX graph capture is unsupported; using direct asynchronous launches"
            logger.warning(self._cuda_graph_error)
        wo.launch(
            self._pipeline,
            dim=(self._render_height, self._render_width),
            inputs=[self._launch_params],
            stream=self._render_stream,
        )

    def _process_debug_output(self):
        self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process_debug(
            self.output_mode,
            self._color_buffer,
            self._depth_buffer,
            self._motion_buffer,
            self._normal_roughness_buffer,
            self._diffuse_buffer,
            self._specular_buffer,
            self._spec_hit_dist_buffer,
            self._render_width,
            self._render_height,
        )

    def _process_final_output(self, source_buffer, *, resize_to_render: bool):
        if resize_to_render:
            self._tonemapper.resize(self._render_width, self._render_height)
        else:
            self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process(source_buffer)

    def _process_output(self, source_buffer, *, resize_final_to_render: bool):
        if self.output_mode == self.OUTPUT_FINAL:
            self._process_final_output(source_buffer, resize_to_render=resize_final_to_render)
            return
        self._process_debug_output()

    def render(self):
        """Render a frame."""
        # Texture transfers and tone mapping use Warp's current stream.
        # Order them with OptiX and denoising, including calls from another stream.
        with wp.ScopedStream(self._render_stream):
            self._render_frame()

    def _render_frame(self):
        if self._pipeline is None:
            logger.error("Pipeline not built. Call build() first.")
            return

        if not self._denoiser_status_reported:
            logger.info(
                "Denoiser active: %s (requested: %s).",
                self.active_denoiser,
                self.denoiser,
            )
            if self.denoiser_error:
                logger.warning("Denoiser fallback: %s", self.denoiser_error)
            self._denoiser_status_reported = True
        self._update_device_camera()

        camera = self._camera_matrices()
        use_external_accum = (
            self.accumulate_samples and self._optix_denoiser is None and self._device_camera_state is None
        )
        samples_this_frame = self.samples_per_frame
        reset_temporal = self._update_temporal_state(camera, use_external_accum)
        self._update_instance_transform_buffers()
        frame = self._launch_samples(samples_this_frame, use_external_accum, camera=camera)

        # Queue the current-to-previous transform snapshot behind the OptiX launch.
        self._snapshot_instance_transforms()

        self._prev_mvp = camera.view_proj
        self._accum_camera_key = camera.key
        self._last_output_mode = self.output_mode

        if self._optix_denoiser is not None:
            output = self._optix_denoiser.apply(
                self._denoiser_inputs(frame),
                reset=reset_temporal or self._denoiser_reset_history,
            )
            self._denoiser_reset_history = False
            self._process_output(output, resize_final_to_render=False)
        else:
            self._process_output(frame, resize_final_to_render=True)
        self.sample_index += samples_this_frame

    def get_output(self) -> np.ndarray:
        """Get the current output as a numpy array."""
        wp.synchronize_stream(self._render_stream)
        return self._tonemapper.get_numpy()

    def resize(self, width: int, height: int):
        """Resize the render buffers."""
        if width != self.width or height != self.height:
            wp.synchronize_stream(self._render_stream)
            # Tear down resources tied to the old dimensions first.  Native
            # window/output resolution remains uncapped; this only avoids a
            # transient old+new allocation spike during resize.
            self._destroy_optix_denoiser()
            self.width = width
            self.height = height
            self.camera.set_aspect_ratio(width, height)
            self._sync_prev_camera_matrices_to_current()
            self._init_denoiser()
            self._tonemapper.resize(width, height)
            self.frame_index = 0

    def close(self):
        """Wait for rendering and release reconstruction resources."""
        wp.synchronize_stream(self._render_stream)
        self._optix_graph_warmed = False
        self._destroy_optix_denoiser()
        # Close explicitly rather than relying on refcounting: other objects
        # (e.g. a UsdScene handle held by the caller) may still reference the scene.
        if self._pipeline is not None:
            self._pipeline.close()
        if self._scene is not None:
            self._scene.close()
        self._device = None
        self._pipeline = None
        self._scene = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def main():
    """Run the path tracing viewer."""
    logger.info("%s", "=" * 60)
    logger.info("OptiX Path Tracing Viewer")
    logger.info("%s", "=" * 60)

    viewer = Renderer(width=800, height=600)

    viewer.build()

    # Render a few frames
    logger.info("Rendering frames.")
    for i in range(10):
        viewer.render()
        logger.info("Frame %d", i + 1)

    # Get final output
    output = viewer.get_output()
    logger.info("Output shape: %s", output.shape)
    logger.info("Output range: [%.3f, %.3f]", float(output.min()), float(output.max()))

    # Save to file if possible
    if Image is not None:
        img_data = (output[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save("pathtracing_output.png")
        logger.info("Saved output to pathtracing_output.png")
    else:
        logger.info("Pillow not installed; skipping image save.")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
