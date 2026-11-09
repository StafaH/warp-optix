# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone path-tracing viewer with a renderer-neutral scene logging API."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from warp_optix.gl import GLInteropViewer

from .camera import cross3
from .defaults import DEFAULT_VIEWER_HEIGHT, DEFAULT_VIEWER_WIDTH
from .pathtracer_api import PathTracer

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _pack_display_rgba8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint32),
    width: int,
    height: int,
):
    y, x = wp.tid()
    if x >= width or y >= height:
        return

    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * width + x] = (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


def _system_pictures_dir() -> Path:
    executable = shutil.which("xdg-user-dir")
    if executable is not None:
        try:
            result = subprocess.run(
                [executable, "PICTURES"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.stdout.strip():
                return Path(result.stdout.strip()).expanduser()
        except (OSError, subprocess.SubprocessError):
            pass
    return Path.home() / "Pictures"


@dataclass
class _InstanceBatch:
    mesh_name: str
    mesh_id: int
    instance_ids: list[int]
    colors: np.ndarray
    materials: np.ndarray
    hidden: bool = False
    active_count: int = 0
    device_material_ids: list[int] = field(default_factory=list)


def _as_numpy(value: Any, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, wp.array):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _broadcast_rows(value: np.ndarray | None, count: int, width: int, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.tile(np.asarray(default, dtype=np.float32), (count, 1))
    rows = np.asarray(value, dtype=np.float32).reshape(-1, width)
    if len(rows) == count:
        return rows.copy()
    if len(rows) == 1:
        return np.repeat(rows, count, axis=0)
    raise ValueError(f"Expected 1 or {count} rows with width {width}, got {len(rows)}")


class ViewerBackend:
    """OptiX viewer backend independent of any simulation framework.

    The ``log_*`` surface intentionally accepts Warp arrays and mirrors the
    renderer-facing protocol used by simulation viewers. It can be used
    directly or placed before a framework's viewer base in an MRO.
    """

    def __init__(
        self,
        width: int = DEFAULT_VIEWER_WIDTH,
        height: int = DEFAULT_VIEWER_HEIGHT,
        title: str = "Warp OptiX Path Tracing",
        fps: int = 0,
        device: str = "cuda",
        headless: bool = False,
        paused: bool = False,
        render_when_paused: bool = False,
        num_frames: int | None = None,
        enable_set: bool = True,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        up_axis: str | int = "Y",
        camera_speed: float = 4.0,
        api: PathTracer | None = None,
        vsync: bool = False,
        max_instances: int = 10000,
        enable_imgui: bool = True,
        enable_picking: bool = True,
        picking_factory: Callable | None = None,
        fallback_to_copy: bool = True,
        default_ior: float = 1.5,
        default_specular: float = 1.0,
        default_clearcoat: float = 0.0,
        default_clearcoat_roughness: float = 0.1,
        enable_texture_mipmaps: bool = False,
        backface_culling: bool = True,
        denoiser: str | None = None,
        optix_upscale: bool = False,
    ):
        try:
            super().__init__()
        except TypeError:
            # A structurally compatible base may require its own explicit
            # initialization in the small integration wrapper.
            pass

        self.width = int(width)
        self.height = int(height)
        self.device = wp.get_device(device)
        self.fps = max(0, int(fps))
        self.headless = bool(headless)
        self.paused = bool(paused)
        self.render_when_paused = bool(render_when_paused)
        self.num_frames = num_frames
        self.frame_index = 0
        self.time = 0.0
        self._closed = False
        self._initialized = False
        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False
        self._last_wall_time = time.perf_counter()
        self._pack_launch = None
        self._warned_texture = False
        self._warned_transform_vbo = False
        self.model = getattr(self, "model", None)
        self.picking_enabled = bool(enable_picking)
        self._picking_factory = picking_factory
        self._picking = None
        self._last_state = None
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._mouse_buttons: set[int] = set()

        self._max_instances = max(1, int(max_instances))
        self._ui_callbacks: list[tuple[Callable, str]] = []
        self._imgui_enabled = bool(enable_imgui)
        self._imgui = None
        self._imgui_impl = None
        self._debug_buffer_mode = 0
        self._current_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.perf_counter()

        self.screenshot_output_path: str | None = None
        self._last_screenshot_path: Path | None = None

        self._api = api or PathTracer(
            width=self.width,
            height=self.height,
            denoiser=denoiser,
            optix_upscale=optix_upscale,
            enable_set=enable_set,
            samples_per_frame=samples_per_frame,
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            enable_texture_mipmaps=enable_texture_mipmaps,
            backface_culling=backface_culling,
        )
        self._presenter = None
        if not self.headless:
            self._presenter = GLInteropViewer(
                width=self.width,
                height=self.height,
                device=str(self.device),
                title=title,
                fps=self.fps,
                on_resize=self._on_resize,
                on_draw_overlay=self._draw_imgui,
                vsync=vsync,
                fallback_to_copy=fallback_to_copy,
            )
            self._presenter.window.push_handlers(self)
            if self._imgui_enabled:
                self._init_imgui()

        self._mesh_ids: dict[str, int] = {}
        self._mesh_texture_ids: dict[str, int] = {}
        self._batches: dict[str, _InstanceBatch] = {}
        self._material_ids: dict[tuple[float, ...], int] = {}
        self._texture_ids: dict[tuple, int] = {}
        self._device_transform_batches: dict[str, tuple[wp.array, wp.array, wp.array]] = {}
        self._device_material_batches: dict[str, tuple[wp.array, wp.array, wp.array]] = {}
        self._default_color = (0.8, 0.8, 0.8)
        self._default_material = (0.5, 0.0, 0.0, 0.0)

        self._default_ior = max(1.0, float(default_ior))
        self._default_specular = float(np.clip(default_specular, 0.0, 1.0))
        self._default_clearcoat = float(np.clip(default_clearcoat, 0.0, 1.0))
        self._default_clearcoat_roughness = float(np.clip(default_clearcoat_roughness, 0.001, 1.0))
        self._camera_position = np.array((0.0, 2.0, 8.0), dtype=np.float32)
        self._camera_yaw = 180.0
        self._camera_pitch = 0.0
        self._camera_fov = 45.0
        self._camera_speed = max(0.0, float(camera_speed))
        self._look_sensitivity = 0.1
        self._keys_down: set[int] = set()
        self._user_camera_control = False
        self._camera_override_this_frame = False
        self._device_camera_bound = False
        self._global_transform = np.eye(4, dtype=np.float32)
        self._up_axis = 1
        self.set_up_axis(up_axis)

    @staticmethod
    def srgb_to_linear(channel: float) -> float:
        """Convert one sRGB channel to linear light."""
        channel = float(np.clip(channel, 0.0, 1.0))
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    @classmethod
    def srgb_to_linear_rgb(cls, color) -> tuple[float, float, float]:
        """Convert an authored sRGB color to path-tracer linear RGB."""
        return tuple(cls.srgb_to_linear(channel) for channel in color[:3])

    @property
    def tonemap_exposure(self) -> float:
        """Return the linear exposure multiplier."""
        return self._api.tonemap_exposure

    @tonemap_exposure.setter
    def tonemap_exposure(self, value: float) -> None:
        """Set the nonnegative linear exposure multiplier."""
        self._api.tonemap_exposure = value

    @property
    def auto_exposure_enabled(self) -> bool:
        """Return whether temporal automatic exposure is enabled."""
        return self._api.auto_exposure_enabled

    def configure_auto_exposure(
        self,
        enabled: bool,
        *,
        target_luminance: float | None = None,
        min_ev: float | None = None,
        max_ev: float | None = None,
        brighten_speed: float | None = None,
        darken_speed: float | None = None,
    ) -> None:
        """Configure GPU temporal automatic exposure."""
        self._api.configure_auto_exposure(
            enabled,
            target_luminance=target_luminance,
            min_ev=min_ev,
            max_ev=max_ev,
            brighten_speed=brighten_speed,
            darken_speed=darken_speed,
        )

    @property
    def analytic_light_intensity(self) -> float:
        """Return the global analytic-light intensity multiplier."""
        return self._api.analytic_light_intensity

    @analytic_light_intensity.setter
    def analytic_light_intensity(self, value: float) -> None:
        """Set the nonnegative analytic-light intensity multiplier."""
        self._api.analytic_light_intensity = value

    @property
    def emissive_material_intensity(self) -> float:
        """Return the global emissive-material intensity multiplier."""
        return self._api.emissive_material_intensity

    @emissive_material_intensity.setter
    def emissive_material_intensity(self, value: float) -> None:
        """Set the nonnegative emissive-material intensity multiplier."""
        self._api.emissive_material_intensity = value

    @property
    def tonemap_contrast(self) -> float:
        """Return the display contrast multiplier."""
        return self._api.tonemap_contrast

    @tonemap_contrast.setter
    def tonemap_contrast(self, value: float) -> None:
        """Set the nonnegative display contrast multiplier."""
        self._api.tonemap_contrast = value

    @property
    def tonemap_saturation(self) -> float:
        """Return the display saturation multiplier."""
        return self._api.tonemap_saturation

    @tonemap_saturation.setter
    def tonemap_saturation(self, value: float) -> None:
        """Set the nonnegative display saturation multiplier."""
        self._api.tonemap_saturation = value

    @property
    def denoiser(self) -> str:
        return self._api.denoiser

    @denoiser.setter
    def denoiser(self, value: str) -> None:
        self.set_denoiser(value)

    @property
    def active_denoiser(self) -> str:
        return self._api.active_denoiser

    @property
    def denoiser_error(self) -> str | None:
        return self._api.denoiser_error

    @property
    def optix_upscale(self) -> bool:
        return self._api.optix_upscale

    @optix_upscale.setter
    def optix_upscale(self, value: bool) -> None:
        self.set_denoiser(self.denoiser, optix_upscale=value)

    def set_denoiser(self, denoiser: str, *, optix_upscale: bool | None = None):
        self._api.set_denoiser(denoiser, optix_upscale=optix_upscale)

    def _ui_denoiser_controls(self, imgui):
        imgui.text(f"Denoiser: {self.active_denoiser}")
        modes = ["auto", "optix", "none"]
        changed, selected = imgui.combo("Denoiser", modes.index(self.denoiser), modes)
        if changed:
            try:
                self.set_denoiser(modes[selected])
            except (RuntimeError, ValueError) as exc:
                logger.warning("Could not switch denoiser: %s", exc)
        if self.denoiser in ("optix", "auto"):
            changed, upscale = imgui.checkbox("OptiX 2x Upscaling", self.optix_upscale)
            if changed:
                try:
                    self.optix_upscale = upscale
                except (RuntimeError, ValueError) as exc:
                    logger.warning("Could not change OptiX upscaling: %s", exc)

    @property
    def max_bounces(self) -> int:
        """Return the current maximum path depth."""
        return self._api.max_bounces

    @property
    def direct_light_samples(self) -> int:
        """Return the number of direct-light samples per surface hit."""
        return self._api.direct_light_samples

    @property
    def russian_roulette_start_bounce(self) -> int:
        """Return the first bounce eligible for stochastic termination."""
        return self._api.russian_roulette_start_bounce

    @property
    def samples_per_frame(self) -> int:
        """Return the number of ray samples per frame."""
        return self._api.samples_per_frame

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        russian_roulette_start_bounce: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust path depth and sampling budgets."""
        self._api.set_ray_budget(
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            russian_roulette_start_bounce=russian_roulette_start_bounce,
            samples_per_frame=samples_per_frame,
        )

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._api.initialize()

        # Keep the neutral lighting defaults used by the reference path-tracing example.
        self._api.set_use_procedural_sky(True)
        self._initialized = True
        self._sync_camera()

    def _qualify_name(self, name: str) -> str:
        qualify = getattr(self, "_qualify", None)
        return qualify(name) if callable(qualify) else str(name)

    def set_up_axis(self, up_axis: str | int):
        axis = "XYZ".index(up_axis.upper()) if isinstance(up_axis, str) else int(up_axis)
        if axis not in (0, 1, 2):
            raise ValueError("up_axis must be X, Y, Z, 0, 1, or 2")
        self._up_axis = axis
        if axis == 2:
            # Physics Z-up -> renderer Y-up: (x, y, z) -> (x, z, -y).
            self._global_transform = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        elif axis == 0:
            # Physics X-up -> renderer Y-up: (x, y, z) -> (-y, x, z).
            self._global_transform = np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        else:
            self._global_transform = np.eye(4, dtype=np.float32)

    def set_model(self, model):
        """Cooperatively initialize a model supplied by a viewer base class."""
        if model is not None and hasattr(model, "up_axis"):
            self.set_up_axis(model.up_axis)
        if self._mesh_ids or self._batches:
            self.clear()
        parent = getattr(super(), "set_model", None)
        if callable(parent):
            parent(model)
        else:
            self.model = model
        self._initialize_picking(model)

    def _initialize_picking(self, model):
        self._picking = None
        if not self.picking_enabled or model is None:
            return

        factory = self._picking_factory
        if factory is None:
            return

        try:
            # Excessive damping kicks moving bodies on mouse-down.
            self._picking = factory(model, pick_stiffness=50.0, pick_damping=5.0)
            if hasattr(self._picking, "world_offsets"):
                self._picking.world_offsets = getattr(self, "world_offsets", None)
            if hasattr(self._picking, "visible_worlds_mask"):
                self._picking.visible_worlds_mask = getattr(self, "_visible_worlds_mask", None)
        except (TypeError, ValueError) as error:
            logger.warning("Picking is unavailable: %s", error)

    def set_world_offsets(self, spacing):
        parent = getattr(super(), "set_world_offsets", None)
        if not callable(parent):
            raise TypeError("World offsets require a simulation viewer base")
        result = parent(spacing)
        if self._picking is not None and hasattr(self._picking, "world_offsets"):
            self._picking.world_offsets = getattr(self, "world_offsets", None)
        return result

    def set_visible_worlds(self, worlds):
        parent = getattr(super(), "set_visible_worlds", None)
        if not callable(parent):
            raise TypeError("Visible worlds require a simulation viewer base")
        result = parent(worlds)
        if self._picking is not None and hasattr(self._picking, "visible_worlds_mask"):
            self._picking.visible_worlds_mask = getattr(self, "_visible_worlds_mask", None)
        return result

    def log_state(self, state):
        """Cache simulation state for picking, then use the framework logger."""
        self._last_state = state
        parent = getattr(super(), "log_state", None)
        if callable(parent):
            parent(state)

    def _get_or_create_texture(self, texture) -> int | None:
        if isinstance(texture, (str, Path)):
            key = ("path", str(texture))
        else:
            key = ("array", id(texture))
        if key in self._texture_ids:
            return self._texture_ids[key]

        try:
            if isinstance(texture, (str, Path)):
                import imageio.v3 as iio

                image = np.asarray(iio.imread(texture))
            else:
                image = np.asarray(texture)
            if image.ndim == 2:
                image = image[..., None]
            if image.ndim != 3 or image.shape[2] not in (1, 2, 3, 4):
                raise ValueError(f"expected an HxWx1-4 image, got shape {image.shape}")
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 2:
                image = np.concatenate((np.repeat(image[..., :1], 3, axis=2), image[..., 1:2]), axis=2)
            if image.shape[2] == 3:
                alpha_value = 255 if np.issubdtype(image.dtype, np.integer) else 1.0
                alpha = np.full((*image.shape[:2], 1), alpha_value)
                image = np.concatenate((image, alpha), axis=2)
            image = np.ascontiguousarray(np.flipud(image))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if not self._warned_texture:
                logger.warning("Could not load mesh texture %r: %s", texture, exc)
                self._warned_texture = True
            return None

        texture_id = self._api.scene.texture_count
        self._api.scene.set_gltf_textures([image], srgb_texture_indices={0}, append=texture_id > 0)
        self._texture_ids[key] = texture_id
        return texture_id

    def _create_material(self, color, roughness, metallic, u_subdiv=0.0, v_subdiv=0.0, texture_id=None) -> int:
        linear_color = self.srgb_to_linear_rgb(color)
        if texture_id is None:
            return self._api.create_pbr_material(
                linear_color,
                roughness=roughness,
                metallic=metallic,
                ior=self._default_ior,
                specular=self._default_specular,
                clearcoat=self._default_clearcoat,
                clearcoat_roughness=self._default_clearcoat_roughness,
                u_subdiv=u_subdiv,
                v_subdiv=v_subdiv,
            )
        return self._api.scene.materials.add_gltf_material(
            base_color=(*linear_color, 1.0),
            roughness=roughness,
            metallic=metallic,
            ior=self._default_ior,
            specular=self._default_specular,
            clearcoat=self._default_clearcoat,
            clearcoat_roughness=self._default_clearcoat_roughness,
            base_color_texture={"index": texture_id, "texCoord": 0},
        )

    def _get_or_create_material(self, color, material, texture_id=None) -> int:
        color_key = tuple(round(float(v), 2) for v in color[:3])
        roughness = round(float(np.clip(material[0], 0.0, 1.0)), 3)
        metallic = round(float(np.clip(material[1], 0.0, 1.0)), 3)
        u_subdiv = round(max(float(material[2]), 0.0), 3)
        v_subdiv = round(max(float(material[3]), 0.0), 3)
        key = (
            *color_key,
            roughness,
            metallic,
            u_subdiv,
            v_subdiv,
            self._default_ior,
            self._default_specular,
            self._default_clearcoat,
            self._default_clearcoat_roughness,
            -1 if texture_id is None else texture_id,
        )
        if key in self._material_ids:
            return self._material_ids[key]

        material_id = self._create_material(color_key, roughness, metallic, u_subdiv, v_subdiv, texture_id)
        self._material_ids[key] = material_id
        self._scene_dirty = True
        return material_id

    def log_mesh(
        self,
        name: str,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """Register or update a mesh prototype."""
        del hidden, backface_culling
        self._ensure_initialized()
        name = self._qualify_name(name)
        points_np = np.ascontiguousarray(_as_numpy(points, np.float32).reshape(-1, 3))
        indices_np = np.ascontiguousarray(_as_numpy(indices, np.uint32).reshape(-1, 3))
        normals_np = None if normals is None else np.ascontiguousarray(_as_numpy(normals, np.float32).reshape(-1, 3))
        uvs_np = None if uvs is None else np.ascontiguousarray(_as_numpy(uvs, np.float32).reshape(-1, 2))
        base_color = color or self._default_color
        pbr = (
            self._default_material[0] if roughness is None else roughness,
            self._default_material[1] if metallic is None else metallic,
            0.0,
            0.0,
        )
        texture_id = None if texture is None else self._get_or_create_texture(texture)
        material_id = self._get_or_create_material(base_color, pbr, texture_id)
        if texture_id is None:
            self._mesh_texture_ids.pop(name, None)
        else:
            self._mesh_texture_ids[name] = texture_id

        if name in self._mesh_ids:
            mesh_id = self._mesh_ids[name]
            from .scene import Mesh

            self._api.scene._meshes[mesh_id] = Mesh(points_np, indices_np, normals_np, uvs_np, material_id=material_id)
        else:
            mesh_id = self._api.create_mesh(points_np, indices_np, normals_np, uvs_np, material_id)
            self._mesh_ids[name] = mesh_id
        self._scene_dirty = True

    def _instance_matrices(self, xforms, scales) -> np.ndarray:
        xforms_np = _as_numpy(xforms, np.float32).reshape(-1, 7)
        scales_np = _broadcast_rows(_as_numpy(scales, np.float32), len(xforms_np), 3, (1.0, 1.0, 1.0))
        count = len(xforms_np)
        matrices = np.zeros((count, 4, 4), dtype=np.float32)
        matrices[:, 3, 3] = 1.0
        matrices[:, :3, 3] = xforms_np[:, :3]

        x, y, z, w = (xforms_np[:, index] for index in range(3, 7))
        matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrices[:, 0, 1] = 2.0 * (x * y - w * z)
        matrices[:, 0, 2] = 2.0 * (x * z + w * y)
        matrices[:, 1, 0] = 2.0 * (x * y + w * z)
        matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrices[:, 1, 2] = 2.0 * (y * z - w * x)
        matrices[:, 2, 0] = 2.0 * (x * z - w * y)
        matrices[:, 2, 1] = 2.0 * (y * z + w * x)
        matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        matrices[:, :3, :3] *= scales_np[:, None, :]
        return np.matmul(self._global_transform[None, :, :], matrices)

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms,
        scales,
        colors,
        materials,
        hidden: bool = False,
    ):
        """Create or update an instanced mesh batch."""
        self._ensure_initialized()
        name = self._qualify_name(name)
        mesh = self._qualify_name(mesh)
        if mesh not in self._mesh_ids:
            raise RuntimeError(f"Mesh prototype {mesh!r} has not been logged")

        count = 0 if xforms is None else len(xforms)
        batch = self._batches.get(name)
        if batch is None:
            batch = _InstanceBatch(
                mesh_name=mesh,
                mesh_id=self._mesh_ids[mesh],
                instance_ids=[],
                colors=np.empty((0, 3), dtype=np.float32),
                materials=np.empty((0, 4), dtype=np.float32),
            )
            self._batches[name] = batch
        elif batch.mesh_name != mesh:
            batch.mesh_name = mesh
            batch.mesh_id = self._mesh_ids[mesh]
            for instance_id in batch.instance_ids:
                self._api.scene._instances[instance_id].mesh_index = batch.mesh_id
            self._scene_dirty = True

        previous_count = len(batch.instance_ids)
        while len(batch.instance_ids) < count:
            if len(self._api.scene._instances) >= self._max_instances:
                raise RuntimeError(f"Viewer instance capacity exceeded ({self._max_instances})")
            batch.instance_ids.append(self._api.create_instance(batch.mesh_id))
            self._scene_dirty = True
        instances_added = len(batch.instance_ids) != previous_count

        appearance_update_requested = colors is not None or materials is not None
        device_appearance = (
            (isinstance(colors, wp.array) and colors.device.is_cuda)
            or (isinstance(materials, wp.array) and materials.device.is_cuda)
            or (name in self._device_material_batches and colors is None and materials is None)
        )
        appearance_changed = False
        texture_id = self._mesh_texture_ids.get(mesh)
        if not device_appearance:
            if colors is not None or len(batch.colors) != count:
                updated_colors = _broadcast_rows(_as_numpy(colors, np.float32), count, 3, self._default_color)
                if updated_colors.shape != batch.colors.shape or not np.array_equal(updated_colors, batch.colors):
                    batch.colors = updated_colors
                    appearance_changed = True
            if materials is not None or len(batch.materials) != count:
                updated_materials = _broadcast_rows(_as_numpy(materials, np.float32), count, 4, self._default_material)
                if updated_materials.shape != batch.materials.shape or not np.array_equal(
                    updated_materials, batch.materials
                ):
                    batch.materials = updated_materials
                    appearance_changed = True

        active_ids = batch.instance_ids[:count]
        inactive_ids = batch.instance_ids[count:]
        visibility_changed = instances_added or hidden != batch.hidden or count != batch.active_count
        if visibility_changed:
            self._api.set_instances_visible(active_ids, not hidden)
            self._api.set_instances_visible(inactive_ids, False)
            batch.hidden = bool(hidden)
            batch.active_count = count

        if count == 0:
            # Hide-only updates keep the cached appearance for later reuse.
            if visibility_changed:
                self._transforms_dirty = True
            return

        if xforms is not None:
            if isinstance(xforms, wp.array) and xforms.device.is_cuda:
                if not isinstance(scales, wp.array) or not scales.device.is_cuda:
                    raise ValueError("CUDA transforms require CUDA-resident scales")
                cached = self._device_transform_batches.get(name)
                if cached is None or len(cached[0]) != count:
                    instance_ids = wp.array(active_ids, dtype=wp.int32, device=self.device)
                else:
                    instance_ids = cached[0]
                device_batch = (instance_ids, xforms, scales)
                self._device_transform_batches[name] = device_batch
                if not self._scene_dirty:
                    self._api.set_instance_transform_arrays(instance_ids, xforms, scales, self._global_transform)
            else:
                matrices = self._instance_matrices(xforms, scales)
                self._api.set_instance_transform_matrices(active_ids, matrices)
            self._transforms_dirty = True

        if device_appearance:
            cached_materials = self._device_material_batches.get(name)
            if cached_materials is None or len(cached_materials[0]) != count:
                if not isinstance(colors, wp.array) or not isinstance(materials, wp.array):
                    raise ValueError("The first CUDA appearance update requires colors and materials")
                material_count = len(batch.device_material_ids)
                for instance_id in active_ids[material_count:]:
                    material_id = self._create_material(
                        self._default_color,
                        self._default_material[0],
                        self._default_material[1],
                        texture_id=texture_id,
                    )
                    self._api.set_instance_material(instance_id, material_id)
                    batch.device_material_ids.append(material_id)
                material_ids = wp.array(batch.device_material_ids[:count], dtype=wp.int32, device=self.device)
                if len(batch.device_material_ids) != material_count:
                    self._scene_dirty = True
            else:
                material_ids = cached_materials[0]
                colors = cached_materials[1] if colors is None else colors
                materials = cached_materials[2] if materials is None else materials
            if not isinstance(colors, wp.array) or not isinstance(materials, wp.array):
                raise ValueError("CUDA appearance updates require CUDA colors and materials")
            device_material_batch = (material_ids, colors, materials)
            self._device_material_batches[name] = device_material_batch
            # New instances also create materials, while the compact CUDA table
            # is rebuilt only by build_scene(). The dirty-scene flush reapplies
            # every cached batch after that table has the required capacity.
            if not self._scene_dirty and appearance_update_requested:
                self._api.set_instance_material_arrays(material_ids, colors, materials)

        if not device_appearance and (appearance_changed or instances_added):
            batch.device_material_ids.clear()
            self._device_material_batches.pop(name, None)
            for index, instance_id in enumerate(active_ids):
                material_id = self._get_or_create_material(batch.colors[index], batch.materials[index], texture_id)
                self._api.set_instance_material(instance_id, material_id)
            self._materials_dirty = True

        if visibility_changed:
            self._transforms_dirty = True

    def log_capsules(self, name, mesh, xforms, scales, colors, materials, hidden: bool = False):
        self.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    def update_instance_transforms(self, name: str, xforms, scales=None):
        """Update an existing batch from Warp arrays without resupplying materials."""
        qualified_name = self._qualify_name(name)
        batch = self._batches.get(qualified_name)
        if batch is None:
            raise KeyError(f"Unknown instance batch {qualified_name!r}")
        self.log_instances(
            qualified_name,
            batch.mesh_name,
            xforms,
            scales,
            None,
            None,
            hidden=False,
        )

    def apply_forces(self, state):
        """Apply the optional picking force to a simulation state."""
        self._last_state = state
        if self._picking is not None:
            self._picking._apply_picking_force(state)

    def _flush_scene(self):
        if self._scene_dirty:
            self._api.build_scene()
            for (
                material_ids,
                colors,
                properties,
            ) in self._device_material_batches.values():
                self._api.set_instance_material_arrays(material_ids, colors, properties)
            for instance_ids, xforms, scales in self._device_transform_batches.values():
                self._api.set_instance_transform_arrays(instance_ids, xforms, scales, self._global_transform)
            if self._device_transform_batches:
                self._api.rebuild_tlas()
        elif self._transforms_dirty:
            self._api.rebuild_tlas()

        if self._materials_dirty and not self._scene_dirty:
            material_ids = np.array(
                [
                    instance.material_id if instance.material_id is not None else np.iinfo(np.uint32).max
                    for instance in self._api.scene._instances
                ],
                dtype=np.uint32,
            )
            self._api.scene.set_instance_material_ids_host(material_ids)

        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False

    def begin_frame(self, time_sec: float):
        parent = getattr(super(), "begin_frame", None)
        if callable(parent):
            parent(time_sec)
        self.time = float(time_sec)
        self._camera_override_this_frame = self._device_camera_bound

    def _render_to_mapped_buffer(self, mapped_image: wp.array, _frame_index: int, _elapsed: float):
        self._api.render_frame()
        source = self._api.viewer.tonemapped_output
        # GL may map the pixel buffer at a different address, so its pointer is part of the key.
        key = (id(source), mapped_image.ptr, self.width, self.height)
        if self._pack_launch is None or self._pack_launch[0] != key:
            command = wp.launch(
                _pack_display_rgba8,
                dim=(self.height, self.width),
                inputs=[source, mapped_image, self.width, self.height],
                device=self.device,
                record_cmd=True,
            )
            self._pack_launch = (key, command)
        self._pack_launch[1].launch()

    def end_frame(self):
        if self._closed:
            return
        if self.paused and not self.render_when_paused and self._presenter is not None:
            self._presenter.window.dispatch_events()
        if self.paused and not self.render_when_paused:
            return
        self._ensure_initialized()
        self._flush_scene()
        now = time.perf_counter()
        if not self._camera_override_this_frame:
            self._update_camera_from_input(max(0.0, min(now - self._last_wall_time, 0.1)))
        self._last_wall_time = now
        if self._presenter is None:
            self._api.render_frame()
        else:
            self._presenter.render_once(self._render_to_mapped_buffer)
        self.frame_index += 1
        self._update_fps()
        if self.num_frames is not None and self.frame_index >= self.num_frames:
            self.close()

    def render(self):
        """Render and present one frame."""
        self.begin_frame(self.time)
        self.end_frame()

    def run(self):
        """Run the standalone viewer until its window closes."""
        while self.is_running():
            self.begin_frame(time.perf_counter())
            self.end_frame()

    def is_running(self) -> bool:
        if self._closed:
            return False
        if self.num_frames is not None and self.frame_index >= self.num_frames:
            return False
        return self._presenter is None or self._presenter.is_running()

    def is_paused(self) -> bool:
        return self.paused

    def close(self):
        self._closed = True
        if self._imgui_impl is not None:
            try:
                self._imgui_impl.shutdown()
            except (AttributeError, RuntimeError):
                pass
            self._imgui_impl = None
        self._api.close()
        if self._presenter is not None:
            self._presenter.close()

    def clear(self):
        self._ensure_initialized()
        self._api.clear_scene()
        self._mesh_ids.clear()
        self._mesh_texture_ids.clear()
        self._batches.clear()
        self._device_transform_batches.clear()
        self._device_material_batches.clear()
        self._material_ids.clear()
        self._texture_ids.clear()
        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False

    @property
    def usd_scene(self):
        """Return the retained USD hierarchy from the loaded OptiX scene."""
        return self._api.usd_scene

    def load_scene_from_usd(
        self,
        usd_path: str,
        *,
        clear_existing: bool = True,
        apply_stage_units: bool = True,
        convert_up_axis: bool = True,
        max_texture_size: int | None = None,
        max_texture_memory_bytes: int | None = None,
        strict_sidedness: bool = False,
        load_usd_environment: bool = False,
        usd_environment_scale: float = 1.0,
        enable_emissive_materials: bool = True,
        load_usd_lights: bool = False,
        usd_light_radius: float | None = None,
    ) -> bool:
        """Load a composed USD stage into the OptiX renderer."""
        self._ensure_initialized()
        loaded = self._api.load_scene_from_usd(
            usd_path,
            clear_existing=clear_existing,
            apply_stage_units=apply_stage_units,
            convert_up_axis=convert_up_axis,
            max_texture_size=max_texture_size,
            max_texture_memory_bytes=max_texture_memory_bytes,
            strict_sidedness=strict_sidedness,
            enable_emissive_materials=enable_emissive_materials,
            load_usd_lights=load_usd_lights,
            usd_light_radius=usd_light_radius,
            load_usd_environment=load_usd_environment,
            usd_environment_scale=usd_environment_scale,
        )
        if loaded and clear_existing:
            self._mesh_ids.clear()
            self._mesh_texture_ids.clear()
            self._batches.clear()
            self._device_transform_batches.clear()
            self._device_material_batches.clear()
            self._material_ids.clear()
            self._texture_ids.clear()
            self._scene_dirty = False
            self._transforms_dirty = False
            self._materials_dirty = False
        return bool(loaded)

    def set_camera_look_at(
        self,
        position,
        target,
        *,
        fov: float | None = None,
        renderer_space: bool = False,
        force: bool = False,
    ) -> None:
        """Set a level camera from an eye and target in physics or renderer space."""
        if self._user_camera_control and not force:
            return
        if force:
            self._camera_override_this_frame = True
        position = np.asarray(position, dtype=np.float32).reshape(3)
        target = np.asarray(target, dtype=np.float32).reshape(3)
        if renderer_space:
            renderer_to_physics = self._global_transform[:3, :3].T
            position = renderer_to_physics @ position
            target = renderer_to_physics @ target
        direction = target - position
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            raise ValueError("Camera position and target must differ")
        direction /= norm

        if self._up_axis == 0:
            pitch = math.degrees(math.asin(float(np.clip(direction[0], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[2]), float(direction[1])))
        elif self._up_axis == 2:
            pitch = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        else:
            pitch = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[2]), float(direction[0])))

        self._camera_position = position
        self._camera_pitch = float(np.clip(pitch, -89.0, 89.0))
        self._camera_yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        if fov is not None:
            self._camera_fov = float(np.clip(fov, 5.0, 120.0))
        self._sync_camera()

    def set_camera_look_at_device(
        self,
        positions: wp.array,
        targets: wp.array,
        *,
        fov: float | wp.array = 45.0,
        renderer_space: bool = False,
    ) -> None:
        """Bind CUDA eye and target arrays written by an application graph.

        The first element of each array is consumed directly on the render
        stream, without a device-to-host camera update.
        """
        camera_transform = np.eye(4, dtype=np.float32) if renderer_space else self._global_transform
        world_up = np.zeros(3, dtype=np.float32)
        world_up[self._up_axis] = 1.0
        self._api.bind_device_camera(
            positions,
            targets,
            fov=fov,
            up=world_up,
            camera_transform=camera_transform,
        )
        self._device_camera_bound = True
        self._camera_override_this_frame = True

    def clear_camera_look_at_device(self) -> None:
        """Return camera control to the normal host-driven viewer path."""
        self._api.unbind_device_camera()
        self._device_camera_bound = False
        self._camera_override_this_frame = False

    def set_camera(self, pos, pitch: float, yaw: float):
        """Set a framework-style camera unless interactive control has taken over."""
        if self._user_camera_control:
            return
        self._camera_position = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        self._camera_pitch = float(np.clip(pitch, -89.0, 89.0))
        self._camera_yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        self._sync_camera()

    def _physics_camera_front(self) -> np.ndarray:
        yaw = math.radians(self._camera_yaw)
        pitch = math.radians(self._camera_pitch)
        cp = math.cos(pitch)
        if self._up_axis == 0:
            return np.array(
                (math.sin(pitch), math.cos(yaw) * cp, math.sin(yaw) * cp),
                dtype=np.float32,
            )
        if self._up_axis == 2:
            return np.array(
                (math.cos(yaw) * cp, math.sin(yaw) * cp, math.sin(pitch)),
                dtype=np.float32,
            )
        return np.array((math.cos(yaw) * cp, math.sin(pitch), math.sin(yaw) * cp), dtype=np.float32)

    def _sync_camera(self):
        if not self._initialized:
            return
        rotation = self._global_transform[:3, :3]
        position = rotation @ self._camera_position
        direction = rotation @ self._physics_camera_front()
        self._api.set_camera_look_at(position, position + direction, (0.0, 1.0, 0.0), self._camera_fov)

    def _update_camera_from_input(self, dt: float):
        if self._presenter is None or not self._keys_down:
            return
        key = self._presenter.pyglet.window.key
        front = self._physics_camera_front()
        up = np.zeros(3, dtype=np.float32)
        up[self._up_axis] = 1.0
        right = cross3(front, up)
        norm = float(np.linalg.norm(right))
        if norm > 1.0e-6:
            right /= norm
        move = np.zeros(3, dtype=np.float32)
        if key.W in self._keys_down or key.UP in self._keys_down:
            move += front
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            move -= front
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            move -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            move += right
        if key.Q in self._keys_down:
            move -= up
        if key.E in self._keys_down:
            move += up
        norm = float(np.linalg.norm(move))
        if norm > 1.0e-6:
            speed = self._camera_speed
            if key.LSHIFT in self._keys_down or key.RSHIFT in self._keys_down:
                speed *= 4.0
            self._camera_position += move / norm * speed * dt
            self._user_camera_control = True
            self._sync_camera()

    def on_key_press(self, symbol, _modifiers):
        if self._presenter is None:
            return
        key = self._presenter.pyglet.window.key
        if self._ui_wants_keyboard():
            return
        if symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.ESCAPE:
            self.close()
            return
        elif symbol == key.P:
            self.save_screenshot()
        else:
            debug_keys = {
                key._1: 2,  # Depth
                key._2: 3,  # Motion
                key._3: 4,  # Normals
                key._4: 6,  # Diffuse
                key._5: 7,  # Specular
                key._6: 1,  # Noisy radiance
                key._7: 8,  # Specular hit distance
                key._8: 5,  # Roughness (OptiX-specific extra)
            }
            if symbol in debug_keys:
                mode = debug_keys[symbol]
                self.set_debug_buffer_mode(0 if self._debug_buffer_mode == mode else mode)
            elif symbol in (key._0, key.BACKSPACE):
                self.set_debug_buffer_mode(0)
        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if self._presenter is None:
            return
        if self._ui_wants_mouse():
            return
        mouse = self._presenter.pyglet.window.mouse
        if self._camera_override_this_frame:
            return
        if buttons & mouse.LEFT:
            self._camera_yaw -= float(dx) * self._look_sensitivity
            self._camera_pitch = float(np.clip(self._camera_pitch + float(dy) * self._look_sensitivity, -89.0, 89.0))
            self._user_camera_control = True
            self._sync_camera()
        if buttons & mouse.RIGHT and self.picking_enabled and self._picking is not None and self._picking.is_picking():
            origin, direction = self._get_ray_from_mouse(_x, _y)
            self._picking.update(wp.vec3(*origin), wp.vec3(*direction))

    def on_mouse_press(self, x, y, button, _modifiers):
        self._mouse_buttons.add(button)
        if self._presenter is None or self._ui_wants_mouse():
            return
        mouse = self._presenter.pyglet.window.mouse
        if (
            button == mouse.RIGHT
            and self.picking_enabled
            and self._picking is not None
            and self._last_state is not None
        ):
            origin, direction = self._get_ray_from_mouse(x, y)
            self._picking.pick(self._last_state, wp.vec3(*origin), wp.vec3(*direction))

    def on_mouse_release(self, _x, _y, button, _modifiers):
        self._mouse_buttons.discard(button)
        if self._presenter is None:
            return
        if button == self._presenter.pyglet.window.mouse.RIGHT and self._picking is not None:
            self._picking.release()

    def on_mouse_motion(self, x, y, _dx, _dy):
        self._mouse_x = float(x)
        self._mouse_y = float(y)

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        if self._ui_wants_mouse():
            return
        if self._camera_override_this_frame:
            return
        self._camera_fov = float(np.clip(self._camera_fov - float(scroll_y) * 2.0, 10.0, 120.0))
        self._user_camera_control = True
        self._sync_camera()

    def _get_ray_from_mouse(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Construct a world-space picking ray in the physics coordinate system."""
        width = max(1.0, float(self.width))
        height = max(1.0, float(self.height))
        ndc_x = 2.0 * float(x) / width - 1.0
        ndc_y = 2.0 * float(y) / height - 1.0

        front = self._physics_camera_front()
        front /= max(float(np.linalg.norm(front)), 1.0e-8)
        world_up = np.zeros(3, dtype=np.float32)
        world_up[self._up_axis] = 1.0
        right = cross3(front, world_up)
        right /= max(float(np.linalg.norm(right)), 1.0e-8)
        camera_up = cross3(right, front)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1.0e-8)

        half_height = math.tan(math.radians(self._camera_fov) * 0.5)
        direction = front + right * ndc_x * (width / height) * half_height + camera_up * ndc_y * half_height
        direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
        return self._camera_position.copy(), direction.astype(np.float32)

    def _on_resize(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._api.resize(self.width, self.height)

    def _update_fps(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = now

    def get_fps(self) -> float:
        return float(self._current_fps)

    def register_ui_callback(self, callback: Callable, position: str = "options"):
        """Register an ImGui callback, matching the hybrid viewer API."""
        self._ui_callbacks.append((callback, str(position)))

    def enable_imgui(self, enabled: bool = True):
        self._imgui_enabled = bool(enabled)
        if self._imgui_enabled and self._imgui_impl is None:
            self._init_imgui()

    def _init_imgui(self):
        if self._presenter is None or self._imgui_impl is not None:
            return
        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend
        except ImportError:
            logger.info("ImGui overlay unavailable; install warp_optix[ui] to enable it")
            return

        try:
            imgui.create_context()
            self._imgui_impl = pyglet_backend.create_renderer(self._presenter.window)
            self._imgui = imgui
            imgui.style_colors_dark()
        except (AttributeError, RuntimeError) as error:
            logger.warning("Failed to initialize ImGui overlay: %s", error)
            self._imgui = None
            self._imgui_impl = None

    def _ui_wants_mouse(self) -> bool:
        if self._imgui is None or self._imgui_impl is None:
            return False
        try:
            return bool(self._imgui.get_io().want_capture_mouse)
        except (AttributeError, RuntimeError):
            return False

    def _ui_wants_keyboard(self) -> bool:
        if self._imgui is None or self._imgui_impl is None:
            return False
        try:
            return bool(self._imgui.get_io().want_capture_keyboard)
        except (AttributeError, RuntimeError):
            return False

    def _draw_imgui(self):
        if not self._imgui_enabled or self._imgui_impl is None:
            return
        imgui = self._imgui
        try:
            self._imgui_impl.process_inputs()
            imgui.new_frame()
            imgui.set_next_window_pos((10.0, 10.0), imgui.Cond_.appearing)
            imgui.set_next_window_size((310.0, 0.0), imgui.Cond_.appearing)
            visible = imgui.begin("OptiX Path Tracing Viewer")
            if isinstance(visible, tuple):
                visible = visible[0]
            if visible:
                imgui.text(f"FPS: {self._current_fps:.1f}")
                imgui.text(f"Meshes: {len(self._mesh_ids)}")
                imgui.text(f"Instances: {sum(len(batch.instance_ids) for batch in self._batches.values())}")
                imgui.text(f"Materials: {len(self._material_ids)}")
                self._ui_denoiser_controls(imgui)

                changed, paused = imgui.checkbox("Pause", self.paused)
                if changed:
                    self.paused = paused
                if hasattr(self, "picking_enabled"):
                    changed, picking = imgui.checkbox("Enable Picking", self.picking_enabled)
                    if changed:
                        self.picking_enabled = picking
                        if not picking and self._picking is not None:
                            self._picking.release()

                visualization_flags = (
                    ("show_joints", "Show Joints"),
                    ("show_contacts", "Show Contacts"),
                    ("show_particles", "Show Particles"),
                    ("show_springs", "Show Springs"),
                    ("show_com", "Show Center of Mass"),
                    ("show_collision", "Show Collision"),
                    ("show_visual", "Show Visual"),
                )
                for attribute, label in visualization_flags:
                    if hasattr(self, attribute):
                        changed, value = imgui.checkbox(label, bool(getattr(self, attribute)))
                        if changed:
                            setattr(self, attribute, value)

                if self.model is not None:
                    if hasattr(self.model, "world_count"):
                        imgui.text(f"Worlds: {self.model.world_count}")
                    elif hasattr(self.model, "num_worlds"):
                        imgui.text(f"Worlds: {self.model.num_worlds}")

                mode_names = [
                    "Final",
                    "Radiance",
                    "Depth",
                    "Motion",
                    "Normal",
                    "Roughness",
                    "Diffuse",
                    "Specular",
                    "Specular Hit Distance",
                ]
                changed, mode = imgui.combo("Debug Buffer", self._debug_buffer_mode, mode_names)
                if changed:
                    self.set_debug_buffer_mode(mode)

                changed, fov = imgui.slider_float("FOV", self._camera_fov, 20.0, 120.0)
                if changed:
                    self._camera_fov = float(fov)
                    self._sync_camera()
                imgui.text(
                    "Camera: "
                    f"({self._camera_position[0]:.2f}, "
                    f"{self._camera_position[1]:.2f}, "
                    f"{self._camera_position[2]:.2f})"
                )
                imgui.text(f"Yaw {self._camera_yaw:.1f}  Pitch {self._camera_pitch:.1f}")

                if imgui.button("Save Screenshot"):
                    self.save_screenshot()
                if self._last_screenshot_path is not None:
                    imgui.text(f"Screenshot: {self._last_screenshot_path}")

                for callback, _position in self._ui_callbacks:
                    callback(imgui)
            imgui.end()
            imgui.render()
            self._imgui_impl.render(imgui.get_draw_data())
        except (AttributeError, RuntimeError, TypeError) as error:
            logger.warning("Disabling ImGui overlay after rendering error: %s", error)
            self._imgui_enabled = False

    def save_screenshot(self, output_path: str | Path | None = None) -> str:
        from PIL import Image

        path = Path(
            output_path
            or self.screenshot_output_path
            or (_system_pictures_dir() / "WarpOptixScreenshots" / time.strftime("pathtracing_%Y%m%d_%H%M%S.png"))
        ).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self._api.get_frame_uint8()
        Image.fromarray(np.flipud(frame[..., :3]), mode="RGB").save(path)
        self._last_screenshot_path = path.resolve()
        logger.info("Saved screenshot to %s", self._last_screenshot_path)
        return str(self._last_screenshot_path)

    def get_instance_transform_gl_buffer(self) -> int:
        """Compatibility API; OptiX uses CUDA arrays instead of a transform VBO."""
        return 0

    def get_instance_transform_capacity(self) -> int:
        return self._max_instances

    def notify_transforms_updated(self, count: int = -1):
        """Compatibility hook for the old bridge-specific transform VBO API."""
        del count
        if not self._warned_transform_vbo:
            logger.warning("No transform GL buffer is exposed; update Warp transforms with log_instances()")
            self._warned_transform_vbo = True

    def is_gpu_transform_available(self) -> bool:
        """The old Vulkan-backed GL transform buffer is not used by this backend."""
        return False

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self._ensure_initialized()
        direction = self._global_transform[:3, :3] @ np.asarray((x, y, z), dtype=np.float32)
        self._api.set_sun_direction(*direction, intensity=float(intensity))

    def reset_temporal_history(self):
        """Discard temporal denoising history after a discontinuous scene change."""
        self._api.reset_temporal_history()

    def set_volume(self, volume, bounds_min, bounds_max, **kwargs):
        self._ensure_initialized()
        self._api.set_volume(volume, bounds_min, bounds_max, **kwargs)

    def set_sky_parameters(self, **kwargs):
        self._ensure_initialized()
        kwargs = dict(kwargs)
        for name in ("ground_color", "night_color"):
            if name in kwargs:
                kwargs[name] = self.srgb_to_linear_rgb(kwargs[name])
        self._api.set_sky_parameters(**kwargs)

    @staticmethod
    def _sun_direction(elevation_degrees: float, azimuth_degrees: float) -> tuple[float, float, float]:
        elevation = math.radians(elevation_degrees)
        azimuth = math.radians(azimuth_degrees)
        cos_elevation = math.cos(elevation)
        return (
            cos_elevation * math.sin(azimuth),
            math.sin(elevation),
            cos_elevation * math.cos(azimuth),
        )

    def set_time_of_day(self, preset: str):
        """Apply one of the physical-sky presets used by the hybrid viewer."""
        presets = {
            "dawn": (
                5.0,
                90.0,
                1.0,
                4.0,
                0.5,
                1.2,
                (0.3, 0.2, 0.15),
                1.5,
                (0.001, 0.002, 0.005),
                0.8,
                1.5,
                2.0,
            ),
            "morning": (
                30.0,
                120.0,
                1.0,
                1.0,
                0.1,
                1.0,
                (0.35, 0.3, 0.25),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                1.0,
            ),
            "noon": (
                60.0,
                180.0,
                1.0,
                0.0,
                0.0,
                1.0,
                (0.4, 0.4, 0.35),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                0.5,
            ),
            "afternoon": (
                45.0,
                240.0,
                1.0,
                0.5,
                0.05,
                1.0,
                (0.4, 0.35, 0.3),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                1.0,
            ),
            "sunset": (
                8.0,
                270.0,
                1.0,
                5.0,
                0.6,
                1.3,
                (0.35, 0.2, 0.1),
                2.0,
                (0.001, 0.002, 0.005),
                0.7,
                2.0,
                3.0,
            ),
            "dusk": (
                -5.0,
                280.0,
                1.0,
                3.0,
                0.3,
                1.1,
                (0.15, 0.1, 0.1),
                2.0,
                (0.005, 0.008, 0.015),
                0.3,
                1.5,
                2.0,
            ),
            "night": (
                -30.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.5,
                (0.025, 0.03, 0.045),
                1.0,
                (0.002, 0.003, 0.006),
                1.0,
                1.0,
                0.15,
            ),
            "midnight": (
                -60.0,
                0.0,
                0.005,
                0.0,
                0.0,
                0.3,
                (0.015, 0.02, 0.035),
                1.0,
                (0.001, 0.0015, 0.003),
                1.0,
                1.0,
                0.08,
            ),
        }
        name = str(preset).lower()
        if name == "midday":
            name = "noon"
        if name not in presets:
            raise ValueError(f"Unknown sky preset {preset!r}; expected one of {', '.join(presets)}")
        (
            elevation,
            azimuth,
            multiplier,
            haze,
            shift,
            saturation,
            ground,
            blur,
            night,
            disk,
            scale,
            glow,
        ) = presets[name]
        self._api.set_sky_parameters(
            sun_direction=self._sun_direction(elevation, azimuth),
            multiplier=multiplier,
            haze=haze,
            red_blue_shift=shift,
            saturation=saturation,
            ground_color=self.srgb_to_linear_rgb(ground),
            horizon_blur=blur,
            night_color=self.srgb_to_linear_rgb(night),
            sun_disk_intensity=disk,
            sun_disk_scale=scale,
            sun_glow_intensity=glow,
            y_is_up=1,
        )

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self._api.set_environment_hdr(hdr_path, scaling)

    def set_environment_color(self, color):
        self._api.set_environment_color(self.srgb_to_linear_rgb(color))

    def set_debug_buffer_mode(self, mode: int):
        mode = int(np.clip(mode, 0, 8))
        self._debug_buffer_mode = mode
        self._api.set_debug_buffer_mode(mode)


class Viewer(ViewerBackend):
    """Concrete standalone path-tracing viewer."""
