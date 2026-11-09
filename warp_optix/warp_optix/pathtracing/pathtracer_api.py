# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""High-level OptiX API for standalone path tracing usage."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import warp as wp

from .arrows import (
    ArrowBatch,
    arrow_segment_indices,
    expand_arrow_material_ids_device_count,
    expand_arrow_material_ids_host_count,
    fill_arrow_curve_buffers,
    update_arrow_curves_device_count,
    update_arrow_curves_host_count,
)
from .defaults import DEFAULT_VIEWER_HEIGHT, DEFAULT_VIEWER_WIDTH
from .pathtracing_viewer import Renderer
from .scene import Curve, Mesh
from .transform_utils import build_transform_matrix

logger = logging.getLogger(__name__)


def _validate_material_ids(scene, material_ids: np.ndarray | None):
    if material_ids is None:
        return None
    values = np.asarray(material_ids)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("material_ids must contain integers")
    if np.any(values < 0) or np.any(values >= scene.materials.count):
        raise ValueError(f"material_ids must reference the {scene.materials.count} existing materials")
    return values


class PathTracer:
    """High-level API for driving the OptiX path tracer directly from Python."""

    def __init__(
        self,
        width: int = DEFAULT_VIEWER_WIDTH,
        height: int = DEFAULT_VIEWER_HEIGHT,
        enable_set: bool = True,
        enable_cuda_graphs: bool = True,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        russian_roulette_start_bounce: int = 3,
        enable_texture_mipmaps: bool = False,
        backface_culling: bool = True,
        denoiser: str | None = None,
        optix_upscale: bool = False,
    ):
        self.width = int(width)
        self.height = int(height)
        self._viewer = Renderer(
            width=self.width,
            height=self.height,
            scene_setup=lambda _scene: None,
            enable_set=bool(enable_set),
            enable_cuda_graphs=bool(enable_cuda_graphs),
            accumulate_samples=False,
            samples_per_frame=samples_per_frame,
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            russian_roulette_start_bounce=russian_roulette_start_bounce,
            enable_texture_mipmaps=enable_texture_mipmaps,
            backface_culling=backface_culling,
            denoiser=denoiser,
            optix_upscale=optix_upscale,
        )
        self._built = False
        self._running = True

    @property
    def viewer(self) -> Renderer:
        return self._viewer

    @property
    def scene(self):
        return self._viewer._scene  # internal, initialized by build()

    @property
    def usd_scene(self):
        """Retained path-addressable hierarchy from the last USD load."""
        scene = self.scene
        return None if scene is None else scene.usd_scene

    @property
    def denoiser(self) -> str:
        """Requested backend: 'auto', 'optix', or 'none'."""
        return self._viewer.denoiser

    @property
    def active_denoiser(self) -> str:
        """Backend currently in use, or 'none' before initialization."""
        return self._viewer.active_denoiser

    @property
    def denoiser_error(self) -> str | None:
        """Initialization diagnostic when automatic selection falls back."""
        return self._viewer.denoiser_error

    @property
    def optix_upscale(self) -> bool:
        """Whether OptiX should denoise and upscale by 2x."""
        return self._viewer.optix_upscale

    def set_denoiser(self, denoiser: str, *, optix_upscale: bool | None = None):
        """Switch backend while retaining each backend's quality settings."""
        self._viewer.set_denoiser(denoiser, optix_upscale=optix_upscale)

    @property
    def texture_mipmaps_enabled(self) -> bool:
        """Return whether texture mip-chain generation is enabled."""
        return bool(self._viewer.enable_texture_mipmaps)

    @property
    def cuda_graph_active(self) -> bool:
        """Whether the stable OptiX launch has been captured for replay."""
        return self._viewer._optix_launch_graph is not None

    @property
    def max_bounces(self) -> int:
        """Return the current maximum path depth."""
        return int(self._viewer.max_bounces)

    @property
    def max_compiled_bounces(self) -> int:
        """Return the largest runtime path depth supported by this pipeline."""
        return int(self._viewer._pipeline_max_bounces)

    @property
    def direct_light_samples(self) -> int:
        """Return the direct-light samples evaluated at each surface hit."""
        return int(self._viewer.direct_light_samples)

    @property
    def russian_roulette_start_bounce(self) -> int:
        """Return the first bounce eligible for stochastic termination."""
        return int(self._viewer.russian_roulette_start_bounce)

    @property
    def samples_per_frame(self) -> int:
        return int(self._viewer.samples_per_frame)

    @property
    def backface_culling(self) -> bool:
        """Whether all path-tracing rays cull back-facing triangles."""
        return bool(self._viewer.backface_culling)

    def set_backface_culling(self, enabled: bool) -> None:
        """Globally enable or disable triangle backface culling at runtime."""
        self._viewer.set_backface_culling(enabled)

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        russian_roulette_start_bounce: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust path depth and sampling budgets."""
        self._viewer.set_ray_budget(
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            russian_roulette_start_bounce=russian_roulette_start_bounce,
            samples_per_frame=samples_per_frame,
        )

    @property
    def linear_depth_output(self):
        """Return the current positive view-space depth buffer."""
        return self._viewer.linear_depth_output

    @property
    def render_resolution(self) -> tuple[int, int]:
        """Return the internal render resolution as ``(width, height)``."""
        return self._viewer.render_resolution

    @property
    def tonemap_exposure(self) -> float:
        """Return the linear exposure multiplier."""
        return float(self._viewer._tonemapper.exposure)

    @tonemap_exposure.setter
    def tonemap_exposure(self, value: float) -> None:
        """Set the nonnegative linear exposure multiplier."""
        self._viewer._tonemapper.exposure = max(0.0, float(value))

    @property
    def auto_exposure_enabled(self) -> bool:
        """Return whether temporal automatic exposure is enabled."""
        return bool(self._viewer._tonemapper.auto_exposure)

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
        self._viewer._tonemapper.configure_auto_exposure(
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
        return float(self._viewer.analytic_light_intensity)

    @analytic_light_intensity.setter
    def analytic_light_intensity(self, value: float) -> None:
        """Set the nonnegative analytic-light intensity multiplier."""
        self._viewer.analytic_light_intensity = max(0.0, float(value))
        self.reset_temporal_history()

    @property
    def emissive_material_intensity(self) -> float:
        """Return the global emissive-material intensity multiplier."""
        return float(self._viewer.emissive_material_intensity)

    @emissive_material_intensity.setter
    def emissive_material_intensity(self, value: float) -> None:
        """Set the nonnegative emissive-material intensity multiplier."""
        self._viewer.emissive_material_intensity = max(0.0, float(value))
        self.reset_temporal_history()

    @property
    def tonemap_contrast(self) -> float:
        """Return the display contrast multiplier."""
        return float(self._viewer._tonemapper.contrast)

    @tonemap_contrast.setter
    def tonemap_contrast(self, value: float) -> None:
        """Set the nonnegative display contrast multiplier."""
        self._viewer._tonemapper.contrast = max(0.0, float(value))

    @property
    def tonemap_saturation(self) -> float:
        """Return the display saturation multiplier."""
        return float(self._viewer._tonemapper.saturation)

    @tonemap_saturation.setter
    def tonemap_saturation(self, value: float) -> None:
        """Set the nonnegative display saturation multiplier."""
        self._viewer._tonemapper.saturation = max(0.0, float(value))

    def _require_scene(self):
        if self._viewer._scene is None:
            self.initialize()
            if self._viewer._scene is None:
                raise RuntimeError("The path tracer is closed")
        return self._viewer._scene

    def initialize(self) -> None:
        """Build the renderer once; failures raise."""
        if not self._built:
            self._viewer.build()
            self._built = True

    def is_running(self) -> bool:
        return self._running

    def close(self):
        self._running = False
        self._viewer.close()

    def begin_frame(self, time_sec: float):
        # Kept for ViewerBase API compatibility; rendering is driven explicitly.
        _ = float(time_sec)

    def end_frame(self):
        # Rendering is explicit in render_frame(); kept for API parity.
        return None

    def render_frame(self):
        self.initialize()
        self._viewer.render()

    def resize(self, width: int, height: int):
        self.initialize()
        self.width = int(width)
        self.height = int(height)
        self._viewer.resize(self.width, self.height)

    def get_frame(self) -> np.ndarray:
        self.initialize()
        return self._viewer.get_output()

    def get_frame_uint8(self) -> np.ndarray:
        image = np.clip(self.get_frame(), 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)

    def build_scene(self):
        scene = self._require_scene()
        scene.build()
        self._viewer.sample_index = 0
        self._viewer.frame_index = 0
        self._viewer._denoiser_reset_history = True
        self._viewer._prev_instance_transforms_valid = False
        self._viewer._sync_prev_camera_matrices_to_current()

    def set_volume(
        self,
        volume: wp.Volume | None,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        **kwargs,
    ):
        """Set or clear the NanoVDB volume rendered by the viewer."""
        self._viewer.set_volume(volume, bounds_min, bounds_max, **kwargs)

    def reset_temporal_history(self):
        """Discard reconstruction history before a discontinuous scene change."""
        self._viewer._denoiser_reset_history = True

    def rebuild_tlas(self):
        scene = self._require_scene()
        scene.rebuild_tlas()

    def clear_scene(self):
        self._require_scene().clear()

    def load_scene_from_gltf(
        self,
        gltf_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        build_scene: bool = True,
    ) -> bool:
        ok = bool(
            self._require_scene().load_from_gltf(
                gltf_path,
                root_transform=root_transform,
                clear_existing=clear_existing,
            )
        )
        if ok and build_scene:
            self.build_scene()
        return ok

    def load_scene_from_usd(
        self,
        usd_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        build_scene: bool = True,
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
        """Load USD geometry and PBR materials through the optional OpenUSD bindings."""
        ok = bool(
            self._require_scene().load_from_usd(
                usd_path,
                root_transform=root_transform,
                clear_existing=clear_existing,
                apply_stage_units=apply_stage_units,
                convert_up_axis=convert_up_axis,
                max_texture_size=max_texture_size,
                max_texture_memory_bytes=max_texture_memory_bytes,
                strict_sidedness=strict_sidedness,
                enable_emissive_materials=enable_emissive_materials,
                load_usd_lights=load_usd_lights,
                usd_light_radius=usd_light_radius,
            )
        )
        if ok and load_usd_environment:
            environment_path = self._require_scene().usd_environment_path
            if environment_path is None:
                logger.warning(
                    "USD stage contains no supported DomeLight environment texture: %s",
                    usd_path,
                )
            else:
                self.set_environment_hdr(str(environment_path), scaling=float(usd_environment_scale))
        if ok and build_scene:
            self.build_scene()
        return ok

    def load_scene_from_obj(self, obj_path: str) -> bool:
        ok = bool(self._require_scene().load_from_obj(obj_path))
        if ok:
            self.build_scene()
        return ok

    def create_mesh(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        material_id: int = 0,
        material_ids: np.ndarray | None = None,
        dynamic: bool = False,
    ) -> int:
        scene = self._require_scene()
        if scene.materials.count == 0:
            scene.materials.add_diffuse((0.8, 0.8, 0.8))
        mat_id = int(material_id)
        if mat_id < 0 or mat_id >= scene.materials.count:
            mat_id = 0
        mesh = Mesh(
            vertices=np.asarray(positions, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.uint32),
            normals=None if normals is None else np.asarray(normals, dtype=np.float32),
            texcoords=None if uvs is None else np.asarray(uvs, dtype=np.float32),
            material_id=mat_id,
            material_ids=_validate_material_ids(scene, material_ids),
            dynamic=dynamic,
        )
        return int(scene.add_mesh(mesh))

    def _dynamic_triangle_mesh(self, geometry_id: int) -> Mesh:
        scene = self._require_scene()
        geometry_id = int(geometry_id)
        if geometry_id < 0 or geometry_id >= scene.geometry_count:
            raise ValueError("mesh geometry ID is out of range")
        mesh = scene._meshes[geometry_id]
        if mesh.primitive_type != "triangles" or not mesh.dynamic:
            raise ValueError("geometry must be a dynamic triangle mesh")
        return mesh

    def get_mesh_vertices_device(self, geometry_id: int) -> wp.array:
        """Return writable CUDA ``wp.vec3`` positions for a dynamic mesh."""
        mesh = self._dynamic_triangle_mesh(geometry_id)
        if mesh.d_vertices is None:
            raise RuntimeError("build the scene before accessing device vertices")
        return wp.array(
            ptr=mesh.d_vertices.ptr,
            shape=(len(mesh.vertices),),
            dtype=wp.vec3,
            device=mesh.d_vertices.device,
        )

    def get_mesh_normals_device(self, geometry_id: int) -> wp.array:
        """Return writable CUDA ``wp.vec3`` normals for a dynamic mesh."""
        mesh = self._dynamic_triangle_mesh(geometry_id)
        if mesh.d_normals is None:
            raise RuntimeError("build the scene before accessing device normals")
        return wp.array(
            ptr=mesh.d_normals.ptr,
            shape=(len(mesh.normals),),
            dtype=wp.vec3,
            device=mesh.d_normals.device,
        )

    def update_mesh_device(
        self,
        geometry_id: int,
        *,
        stream=None,
        rebuild_gas: bool = False,
        rebuild_tlas: bool = True,
    ) -> None:
        """Update a dynamic triangle GAS after its CUDA buffers were modified."""
        self._dynamic_triangle_mesh(geometry_id)
        self._require_scene().update_mesh_accel(
            geometry_id,
            stream=stream,
            rebuild_gas=bool(rebuild_gas),
            rebuild_tlas=bool(rebuild_tlas),
        )

    def create_curve(
        self,
        positions: np.ndarray,
        radii: np.ndarray | float,
        segment_indices: np.ndarray | None = None,
        material_id: int = 0,
        material_ids: np.ndarray | None = None,
        basis: str = "linear",
        dynamic: bool = False,
    ) -> int:
        """Create reusable native round linear or cubic Bézier curve geometry.

        Args:
            positions: ``(N, 3)`` control points.
            radii: Positive scalar radius or one radius per control point.
            segment_indices: Optional start control-point index per segment.
                Linear curves default to all consecutive pairs. Cubic Bézier
                curves default to starts ``0, 3, 6, ...`` for ``3*N+1`` points.
                Explicit indices can pack multiple disjoint strands.
            basis: ``"linear"`` or ``"cubic_bezier"``.
            material_id: Existing PBR material applied to every segment.
            material_ids: Optional material-table index per segment. When
                supplied, these assignments take precedence over material_id.

        The returned geometry ID works with the same instance, transform,
        visibility, and material-override methods as a mesh ID.
        """
        scene = self._require_scene()
        if scene.materials.count == 0:
            scene.materials.add_diffuse((0.8, 0.8, 0.8))
        mat_id = int(material_id)
        if mat_id < 0 or mat_id >= scene.materials.count:
            mat_id = 0
        curve = Curve(
            positions,
            radii,
            segment_indices,
            material_id=mat_id,
            material_ids=_validate_material_ids(scene, material_ids),
            basis=basis,
            dynamic=dynamic,
        )
        return int(scene.add_curve(curve))

    def update_curve_device(
        self,
        geometry_id: int,
        positions: wp.array,
        *,
        radii: wp.array | None = None,
        material_ids: wp.array | None = None,
        stream=None,
        rebuild_gas: bool = False,
        rebuild_tlas: bool = True,
    ) -> None:
        """Update a dynamic curve from CUDA-resident flat buffers."""
        scene = self._require_scene()
        curve = scene._meshes[int(geometry_id)]
        if curve.primitive_type == "triangles" or not curve.dynamic:
            raise ValueError("geometry must be a dynamic curve")
        if curve.d_vertices is None:
            raise RuntimeError("build the scene before updating curves")
        if not positions.device.is_cuda or positions.dtype != wp.float32:
            raise ValueError("positions must be a CUDA float32 Warp array")
        if positions.ndim != 1 or len(positions) != len(curve.d_vertices):
            raise ValueError("positions must contain three floats per control point")
        wp.copy(curve.d_vertices, positions, stream=stream)
        if radii is not None:
            if not radii.device.is_cuda or radii.dtype != wp.float32:
                raise ValueError("radii must be a CUDA float32 Warp array")
            if radii.ndim != 1 or len(radii) != len(curve.d_widths):
                raise ValueError("radii must contain one value per control point")
            wp.copy(curve.d_widths, radii, stream=stream)
        if material_ids is not None:
            if not material_ids.device.is_cuda or material_ids.dtype != wp.uint32:
                raise ValueError("material_ids must be a CUDA uint32 Warp array")
            if material_ids.ndim != 1 or len(material_ids) != len(curve.material_ids):
                raise ValueError("material_ids must contain one value per segment")
            wp.copy(
                scene._packed_material_ids,
                material_ids,
                dest_offset=curve._packed_material_offset,
                stream=stream,
            )
        scene.update_curve_accel(
            int(geometry_id),
            stream=stream,
            rebuild_gas=bool(rebuild_gas),
            rebuild_tlas=bool(rebuild_tlas),
        )

    def create_arrow_batch(
        self,
        capacity: int,
        small_radius: float,
        large_radius: float,
        tip_length_ratio: float = 0.2,
        material_id: int = 0,
        material_ids: np.ndarray | None = None,
    ) -> ArrowBatch:
        """Create one fixed-capacity, dynamically refittable arrow batch.

        The implementation stores every arrow as two native round-linear curve
        primitives in one GAS: a constant-radius shaft and a linearly tapered
        tip. ``tip_length_ratio`` is the fraction of total arrow length occupied
        by the tip. ``material_ids``, when supplied, contains one material per
        arrow; the shaft and tip share it.

        Choose ``capacity`` as the simulation's maximum contact count. Runtime
        updates may freely change the active count up to that capacity without
        reallocating geometry or rebuilding the scene.
        """
        scene = self._require_scene()
        capacity = int(capacity)
        small_radius = float(small_radius)
        large_radius = float(large_radius)
        tip_length_ratio = float(tip_length_ratio)
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if small_radius <= 0.0 or large_radius <= 0.0:
            raise ValueError("small_radius and large_radius must be positive")
        if large_radius < small_radius:
            raise ValueError("large_radius must be at least small_radius")
        if not 0.0 < tip_length_ratio < 1.0:
            raise ValueError("tip_length_ratio must be between zero and one")

        if scene.materials.count == 0:
            scene.materials.add_diffuse((0.8, 0.8, 0.8))
        material_id = int(material_id)
        if material_id < 0 or material_id >= scene.materials.count:
            material_id = 0
        if material_ids is None:
            segment_material_ids = np.full(2 * capacity, material_id, dtype=np.uint32)
        else:
            arrow_material_ids = _validate_material_ids(scene, material_ids)
            if arrow_material_ids.shape != (capacity,):
                raise ValueError(f"material_ids must have shape ({capacity},)")
            segment_material_ids = np.repeat(np.asarray(arrow_material_ids, dtype=np.uint32), 2)

        curve = Curve(
            np.zeros((4 * capacity, 3), dtype=np.float32),
            np.zeros(4 * capacity, dtype=np.float32),
            arrow_segment_indices(capacity),
            material_id=material_id,
            material_ids=segment_material_ids,
            dynamic=True,
        )
        geometry_id = int(scene.add_curve(curve))
        instance_id = int(scene.add_instance(geometry_id))
        return ArrowBatch(
            geometry_id=geometry_id,
            instance_id=instance_id,
            capacity=capacity,
            small_radius=small_radius,
            large_radius=large_radius,
            tip_length_ratio=tip_length_ratio,
        )

    def _arrow_curve(self, batch: ArrowBatch) -> Curve:
        if not isinstance(batch, ArrowBatch):
            raise TypeError("batch must be an ArrowBatch")
        scene = self._require_scene()
        if batch.geometry_id < 0 or batch.geometry_id >= scene.geometry_count:
            raise ValueError("arrow batch does not belong to this scene")
        curve = scene._meshes[batch.geometry_id]
        if curve.primitive_type != "round_linear" or not curve.dynamic:
            raise ValueError("arrow batch geometry is no longer available")
        return curve

    def update_arrow_batch(
        self,
        batch: ArrowBatch,
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        material_ids: np.ndarray | None = None,
        rebuild_tlas: bool = True,
        rebuild_gas: bool = True,
    ) -> None:
        """Update arrows from host arrays and update the dynamic curve GAS.

        A fast GAS rebuild is the default because contact ordering may shuffle.
        Set ``rebuild_gas=False`` only when arrow identities remain spatially stable.
        """
        scene = self._require_scene()
        curve = self._arrow_curve(batch)
        starts = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
        count = len(starts)
        fill_arrow_curve_buffers(
            starts,
            ends,
            curve.vertices,
            curve.radii,
            batch.small_radius,
            batch.large_radius,
            batch.tip_length_ratio,
        )
        batch.active_count = count

        if material_ids is not None:
            arrow_material_ids = _validate_material_ids(scene, material_ids)
            if arrow_material_ids.shape != (count,):
                raise ValueError(f"material_ids must have shape ({count},)")
            curve.material_ids[: 2 * count] = np.repeat(np.asarray(arrow_material_ids, dtype=np.uint32), 2)

        if curve.d_vertices is None:
            return
        curve.d_vertices.assign(curve.vertices.reshape(-1))
        curve.d_widths.assign(curve.radii)
        if material_ids is not None:
            device_material_ids = wp.array(curve.material_ids, dtype=wp.uint32, device="cuda")
            wp.copy(
                scene._packed_material_ids,
                device_material_ids,
                dest_offset=curve._packed_material_offset,
                count=len(curve.material_ids),
            )
        scene.update_curve_accel(
            batch.geometry_id,
            rebuild_gas=bool(rebuild_gas),
            rebuild_tlas=bool(rebuild_tlas),
        )

    def update_arrow_batch_device(
        self,
        batch: ArrowBatch,
        starts: wp.array,
        ends: wp.array,
        active_count: int | wp.array,
        *,
        material_ids: wp.array | None = None,
        stream=None,
        rebuild_gas: bool = True,
        rebuild_tlas: bool = True,
    ) -> None:
        """Update arrows without CPU readback from CUDA-resident Warp arrays.

        ``starts`` and ``ends`` must be one-dimensional ``wp.vec3`` arrays.
        ``active_count`` may be a host integer or a one-element CUDA ``int32``
        array. The device-count form launches over the full capacity, allowing
        a simulation to change the contact count without synchronizing it to
        the CPU; its value is clamped to the batch capacity on-device. Device
        material IDs are optional and contain one ``int32`` ID per arrow.

        A fast GAS rebuild is the default because contact ordering may shuffle.
        Set ``rebuild_gas=False`` only for spatially stable arrow identities.
        """
        scene = self._require_scene()
        curve = self._arrow_curve(batch)
        if curve.d_vertices is None:
            raise RuntimeError("build the scene before device arrow updates")
        for name, values in (("starts", starts), ("ends", ends)):
            if not hasattr(values, "device") or not values.device.is_cuda:
                raise ValueError(f"{name} must be a CUDA-resident Warp array")
            if values.ndim != 1 or values.dtype != wp.vec3:
                raise ValueError(f"{name} must be a one-dimensional wp.vec3 array")
        if starts.device != ends.device:
            raise ValueError("starts and ends must be on the same CUDA device")
        if material_ids is not None:
            if not material_ids.device.is_cuda or material_ids.dtype != wp.int32:
                raise ValueError("material_ids must be a CUDA int32 Warp array")
            if material_ids.ndim != 1:
                raise ValueError("material_ids must be one-dimensional")

        previous_count = batch.active_count
        device_count = hasattr(active_count, "device")
        if device_count:
            if not active_count.device.is_cuda or active_count.dtype != wp.int32:
                raise ValueError("active_count must be a CUDA int32 Warp array")
            if active_count.ndim != 1 or len(active_count) < 1:
                raise ValueError("active_count must contain at least one value")
            if len(starts) < batch.capacity or len(ends) < batch.capacity:
                raise ValueError("device-count endpoint arrays must contain batch.capacity values")
            if material_ids is not None and len(material_ids) < batch.capacity:
                raise ValueError("device-count material_ids must contain batch.capacity values")
            launch_dim = batch.capacity
            geometry_kernel = update_arrow_curves_device_count
            geometry_inputs = [
                starts,
                ends,
                active_count,
                batch.capacity,
                batch.small_radius,
                batch.large_radius,
                batch.tip_length_ratio,
                curve.d_vertices,
                curve.d_widths,
            ]
            batch.active_count = -1
        else:
            count = int(active_count)
            if count < 0 or count > batch.capacity:
                raise ValueError("active_count must be within batch capacity")
            if len(starts) < count or len(ends) < count:
                raise ValueError("endpoint arrays contain fewer than active_count values")
            if material_ids is not None and len(material_ids) < count:
                raise ValueError("material_ids contains fewer than active_count values")
            launch_dim = batch.capacity if previous_count < 0 else max(count, previous_count)
            geometry_kernel = update_arrow_curves_host_count
            geometry_inputs = [
                starts,
                ends,
                count,
                batch.small_radius,
                batch.large_radius,
                batch.tip_length_ratio,
                curve.d_vertices,
                curve.d_widths,
            ]
            batch.active_count = count

        if launch_dim:
            wp.launch(
                geometry_kernel,
                dim=launch_dim,
                inputs=geometry_inputs,
                device=starts.device,
                stream=stream,
            )
            if material_ids is not None:
                material_kernel = (
                    expand_arrow_material_ids_device_count if device_count else expand_arrow_material_ids_host_count
                )
                material_inputs = [material_ids, active_count]
                if device_count:
                    material_inputs.append(batch.capacity)
                material_inputs.extend([scene._packed_material_ids, curve._packed_material_offset])
                wp.launch(
                    material_kernel,
                    dim=launch_dim,
                    inputs=material_inputs,
                    device=starts.device,
                    stream=stream,
                )
        scene.update_curve_accel(
            batch.geometry_id,
            rebuild_gas=bool(rebuild_gas),
            stream=stream,
            rebuild_tlas=bool(rebuild_tlas),
        )

    def create_instance(self, mesh_id: int) -> int:
        """Create an instance of a mesh or curve geometry ID."""
        return int(self._require_scene().add_instance(int(mesh_id)))

    def set_instance_material(self, instance_id: int, material_id: int | None):
        self._require_scene().set_instance_material(int(instance_id), material_id)

    def set_instance_visible(self, instance_id: int, visible: bool):
        self._require_scene().set_instance_visible(int(instance_id), bool(visible))

    def create_instance_with_transform(
        self,
        mesh_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ) -> int:
        transform = build_transform_matrix(position, rotation_xyzw, scale)
        return int(self._require_scene().add_instance(int(mesh_id), transform=transform))

    def set_instance_transform(
        self,
        instance_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ):
        transform = build_transform_matrix(position, rotation_xyzw, scale)
        self._require_scene().set_instance_transform(int(instance_id), transform)

    def set_instance_transform_matrix(self, instance_id: int, matrix: np.ndarray):
        m = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
        self._require_scene().set_instance_transform(int(instance_id), m)

    def set_instance_transform_matrices(self, instance_ids: Iterable[int], matrices: np.ndarray):
        self._require_scene().set_instance_transforms_batch(instance_ids, matrices)

    def set_instance_transform_arrays(self, instance_ids, xforms, scales, global_transform) -> bool:
        """Update instance transforms from CUDA-resident Warp arrays."""
        return self._require_scene().set_instance_transforms_device(instance_ids, xforms, scales, global_transform)

    def set_instance_material_arrays(self, material_ids, colors, properties) -> bool:
        """Update compact materials from CUDA-resident Warp arrays."""
        return self._require_scene().set_instance_materials_device(material_ids, colors, properties)

    def set_instances_visible(self, instance_ids: Iterable[int], visible: bool):
        self._require_scene().set_instances_visible_batch(instance_ids, visible)

    def set_instance_transforms_batch(self, instance_ids: Iterable[int], transforms_flat: np.ndarray):
        self.initialize()
        ids = list(instance_ids)
        arr = np.asarray(transforms_flat, dtype=np.float32).reshape(-1, 8)
        for i, instance_id in enumerate(ids):
            if i >= len(arr):
                break
            row = arr[i]
            self.set_instance_transform(
                int(instance_id),
                position=row[0:3],
                rotation_xyzw=row[3:7],
                scale=float(row[7]),
            )

    def create_diffuse_material(self, color: Iterable[float]) -> int:
        return int(self._require_scene().materials.add_diffuse(tuple(float(v) for v in color)))

    def create_metallic_material(self, color: Iterable[float], roughness: float = 0.1) -> int:
        return int(self._require_scene().materials.add_metal(tuple(float(v) for v in color), float(roughness)))

    def create_emissive_material(self, color: Iterable[float], intensity: float = 1.0) -> int:
        return int(self._require_scene().materials.add_emissive(tuple(float(v) for v in color), float(intensity)))

    def create_pbr_material(
        self,
        color: Iterable[float],
        roughness: float,
        metallic: float,
        ior: float = 1.5,
        specular: float = 1.0,
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.1,
        u_subdiv: float = 0.0,
        v_subdiv: float = 0.0,
        base_color_scale: float = 0.75,
        emissive: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> int:
        scene = self._require_scene()
        return int(
            scene.materials.add_pbr(
                base_color=tuple(float(v) for v in color),
                roughness=float(roughness),
                metallic=float(metallic),
                ior=max(1.0, float(ior)),
                specular=min(1.0, max(0.0, float(specular))),
                clearcoat=min(1.0, max(0.0, float(clearcoat))),
                clearcoat_roughness=min(1.0, max(0.001, float(clearcoat_roughness))),
                u_subdiv=float(u_subdiv),
                v_subdiv=float(v_subdiv),
                base_color_scale=float(base_color_scale),
                emissive=tuple(float(v) for v in emissive),
            )
        )

    def add_box(self, min_pt: Iterable[float], max_pt: Iterable[float], material_id: int) -> int:
        return int(
            self._require_scene().add_box(
                tuple(float(v) for v in min_pt),
                tuple(float(v) for v in max_pt),
                int(material_id),
            )
        )

    def add_sphere(self, center: Iterable[float], radius: float, segments: int, material_id: int) -> int:
        return int(
            self._require_scene().add_sphere(
                tuple(float(v) for v in center),
                float(radius),
                int(segments),
                int(material_id),
            )
        )

    def set_camera_look_at(
        self,
        position: Iterable[float],
        target: Iterable[float],
        up: Iterable[float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
    ):
        self.initialize()
        self._viewer.camera.position = np.asarray(list(position), dtype=np.float32)
        self._viewer.camera.target = np.asarray(list(target), dtype=np.float32)
        self._viewer.camera.up = np.asarray(list(up), dtype=np.float32)
        self._viewer.camera.fov = float(fov)

    def bind_device_camera(
        self,
        positions,
        targets,
        *,
        fov=45.0,
        up=(0.0, 1.0, 0.0),
        camera_transform=None,
    ):
        """Bind graph-written CUDA eye and target arrays to the path tracer."""
        self.initialize()
        self._viewer.bind_device_camera(positions, targets, fov=fov, up=up, camera_transform=camera_transform)

    def unbind_device_camera(self):
        """Restore the host-driven path tracer camera."""
        self._viewer.unbind_device_camera()

    def set_camera_angles(
        self,
        position: Iterable[float],
        yaw: float,
        pitch: float,
        fov: float = 45.0,
    ):
        self.initialize()
        yaw_rad = np.deg2rad(float(yaw))
        pitch_rad = np.deg2rad(float(pitch))
        direction = np.array(
            [
                np.sin(yaw_rad) * np.cos(pitch_rad),
                np.sin(pitch_rad),
                np.cos(yaw_rad) * np.cos(pitch_rad),
            ],
            dtype=np.float32,
        )
        pos = np.asarray(list(position), dtype=np.float32)
        self.set_camera_look_at(pos, pos + direction, (0.0, 1.0, 0.0), float(fov))

    def set_debug_buffer_mode(self, mode: int):
        self.initialize()
        self._viewer.output_mode = int(mode)

    def set_use_procedural_sky(self, enabled: bool):
        self.initialize()
        if enabled:
            self._viewer._env_map = None

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self.initialize()
        direction = np.array([x, y, z], dtype=np.float32)
        nrm = np.linalg.norm(direction)
        if nrm > 0.0:
            direction = direction / nrm
        self._viewer.sky_sun_direction = (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        )
        self._viewer.sky_multiplier = float(intensity)

    def set_sky_parameters(
        self,
        sun_direction: Iterable[float],
        multiplier: float = 1.0,
        haze: float = 0.0,
        red_blue_shift: float = 0.0,
        saturation: float = 1.0,
        horizon_height: float = 0.0,
        ground_color: Iterable[float] = (0.4, 0.4, 0.4),
        horizon_blur: float = 1.0,
        night_color: Iterable[float] = (0.0, 0.0, 0.0),
        sun_disk_intensity: float = 1.0,
        sun_disk_scale: float = 1.0,
        sun_glow_intensity: float = 1.0,
        y_is_up: int = 1,
        grayscale: float | bool = 0.5,
    ):
        self.initialize()
        direction = np.asarray(tuple(float(v) for v in sun_direction), dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("sun_direction must be nonzero")
        direction /= norm
        self._viewer.sky_sun_direction = tuple(float(v) for v in direction)
        self._viewer.sky_multiplier = float(multiplier)
        self._viewer.sky_haze = float(haze)
        self._viewer.sky_redblueshift = float(red_blue_shift)
        self._viewer.sky_saturation = float(saturation)
        self._viewer.sky_horizon_height = float(horizon_height)
        self._viewer.sky_ground_color = tuple(float(v) for v in ground_color)
        self._viewer.sky_horizon_blur = float(horizon_blur)
        self._viewer.sky_night_color = tuple(float(v) for v in night_color)
        self._viewer.sky_sun_disk_intensity = float(sun_disk_intensity)
        self._viewer.sky_sun_disk_scale = float(sun_disk_scale)
        self._viewer.sky_sun_glow_intensity = float(sun_glow_intensity)
        self._viewer.sky_y_is_up = int(y_is_up)
        self._viewer.sky_grayscale = float(np.clip(float(grayscale), 0.0, 1.0))

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self.initialize()
        self._viewer.set_environment_hdr(hdr_path, scaling=float(scaling))

    def set_environment_color(self, color: Iterable[float]):
        self.initialize()
        self._viewer.set_environment_color(tuple(float(v) for v in color))
