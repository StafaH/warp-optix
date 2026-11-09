# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path trace a user-supplied USD stage with OpenUSD composition and PBR materials."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import warp as wp
from example_warp_optix_basic_pathtracing import FreeCameraController, _pack_display_rgba8
from warp_optix.gl import GLInteropViewer
from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    PathTracer,
)


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("scene_usd", type=Path, help="USD/USDA/USDC/USDZ stage to load.")
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--fps", type=int, default=0, help="Presentation rate cap (0 = unlimited).")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0 = run forever).")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save the final tone-mapped frame as a PNG (use with --max-frames).",
    )
    parser.add_argument("--title", type=str, default="Warp OptiX USD Pathtracing")
    parser.add_argument(
        "--camera-speed", type=float, default=None, help="Movement speed; defaults to 20%% of scene radius."
    )
    parser.add_argument(
        "--usd-camera",
        type=str,
        default=None,
        help="Explicit USD Camera prim path; otherwise use the stage-bound or best authored camera.",
    )
    parser.add_argument(
        "--no-usd-camera",
        action="store_true",
        help="Ignore authored USD cameras and frame the transformed geometry bounds.",
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the initial camera position (requires --camera-target).",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the initial camera target (requires --camera-position).",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=None,
        help="Override the initial vertical field of view in degrees.",
    )
    parser.add_argument(
        "--no-stage-units", action="store_true", help="Keep authored units instead of converting to meters."
    )
    parser.add_argument(
        "--no-up-axis-conversion", action="store_true", help="Keep the stage axis instead of converting to Y-up."
    )
    parser.add_argument(
        "--max-texture-size",
        type=int,
        default=512,
        help="Maximum size of each source texture/UDIM tile; stitched atlases may grow larger (0 keeps full resolution).",
    )
    parser.add_argument(
        "--load-usd-environment",
        action="store_true",
        help="Use the first lat-long DomeLight texture composed by the USD stage.",
    )
    parser.add_argument(
        "--strict-usd-sidedness",
        action="store_true",
        help="Honor authored USD doubleSided metadata (default renders robustly two-sided).",
    )
    parser.add_argument(
        "--usd-environment-scale",
        type=float,
        default=1.0,
        help="Intensity multiplier applied to the opted-in USD environment texture.",
    )
    parser.add_argument("--exposure", type=float, default=0.68, help="Linear display exposure multiplier.")
    parser.add_argument("--contrast", type=float, default=1.08, help="Display contrast multiplier.")
    parser.add_argument("--saturation", type=float, default=1.1, help="Display saturation multiplier.")
    parser.add_argument("--denoiser", choices=("auto", "optix", "none"), default="auto")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable OptiX CUDA graph replay.")
    parser.add_argument("--no-set", action="store_true", help="Disable Shader Execution Reordering.")
    parser.add_argument(
        "--no-backface-culling",
        action="store_true",
        help="Render back-facing triangles for globally two-sided traversal.",
    )
    parser.add_argument(
        "--debug-buffer-mode",
        type=int,
        default=0,
        choices=range(9),
        metavar="0..8",
        help="Initial debug view: 0 final, 1 radiance, 2 depth, 3 motion, 4 normal, "
        "5 roughness, 6 diffuse, 7 specular, 8 specular hit distance.",
    )
    parser.add_argument(
        "--disable-normal-textures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-orm-textures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _world_bounds(api: PathTracer):
    scene = api.scene
    instance_bounds = []
    local_bounds = {}
    for instance_index, instance in enumerate(scene._instances):
        mesh_index = instance.mesh_index
        if mesh_index not in local_bounds:
            vertices = scene._meshes[mesh_index].vertices
            local_bounds[mesh_index] = (np.min(vertices, axis=0), np.max(vertices, axis=0))
        local_min, local_max = local_bounds[mesh_index]
        corners = np.array(
            [
                (x, y, z, 1.0)
                for x in (local_min[0], local_max[0])
                for y in (local_min[1], local_max[1])
                for z in (local_min[2], local_max[2])
            ],
            dtype=np.float64,
        )
        world = scene._instance_transform_cache[instance_index]
        transformed = (world @ corners.T).T[:, :3]
        instance_bounds.append((np.min(transformed, axis=0), np.max(transformed, axis=0)))

    if not instance_bounds:
        return np.full(3, -0.5), np.full(3, 0.5)

    # Physics stages commonly contain a very large, zero-thickness ground
    # plane next to a comparatively small subject. Fitting that plane makes
    # the subject effectively invisible (for example a 16 cm car on a 50 m
    # plane). Ignore only extreme flat support surfaces for initial framing;
    # they remain fully loaded and rendered.
    sizes = np.asarray([maximum - minimum for minimum, maximum in instance_bounds])
    diagonals = np.linalg.norm(sizes, axis=1)
    positive = diagonals[diagonals > 1.0e-12]
    if len(positive):
        ordered = np.sort(positive)
        # Use the lower half so one or two enormous support meshes cannot
        # inflate the reference scale used to identify those same outliers.
        typical = float(np.median(ordered[: max(1, len(ordered) // 2)]))
    else:
        typical = 0.0
    selected = []
    for item, size, diagonal in zip(instance_bounds, sizes, diagonals, strict=True):
        longest = max(float(np.max(size)), 1.0e-20)
        is_flat = float(np.min(size)) <= longest * 1.0e-5
        is_extreme = typical > 0.0 and float(diagonal) > typical * 25.0
        if not (is_flat and is_extreme):
            selected.append(item)
    if not selected:
        selected = instance_bounds

    bounds_min = np.min(np.asarray([item[0] for item in selected]), axis=0)
    bounds_max = np.max(np.asarray([item[1] for item in selected]), axis=0)
    return bounds_min, bounds_max


def _camera_angles(position, direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-20)
    yaw = math.degrees(math.atan2(float(direction[0]), float(direction[2])))
    pitch = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
    return np.asarray(position, dtype=np.float32), yaw, pitch


def _authored_camera(api: PathTracer, requested_path: str | None = None):
    from pxr import UsdGeom, UsdRender

    usd_scene = api.usd_scene
    stage = usd_scene.stage
    camera_paths = []
    if requested_path:
        camera_paths.append(requested_path)
    else:
        settings = UsdRender.Settings.GetStageRenderSettings(stage)
        if settings:
            camera_paths.extend(str(path) for path in settings.GetCameraRel().GetTargets())
        custom_data = stage.GetPseudoRoot().GetMetadata("customLayerData") or {}
        bound_camera = custom_data.get("cameraSettings", {}).get("boundCamera")
        if bound_camera:
            camera_paths.append(str(bound_camera))
        unsuitable_tokens = ("follow", "velocity", "physics", "collision", "debug")
        authored = [
            str(prim.GetPath())
            for prim in stage.TraverseAll()
            if prim.IsA(UsdGeom.Camera)
            and str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible"
            and not any(token in prim.GetName().lower() for token in unsuitable_tokens)
        ]

        def camera_score(path):
            name = Path(path).name.lower()
            return (0 if "overview" in name else 1 if "main" in name else 2, path)

        camera_paths.extend(sorted(authored, key=camera_score))

    seen = set()
    for camera_path in camera_paths:
        if camera_path in seen:
            continue
        seen.add(camera_path)
        prim = stage.GetPrimAtPath(camera_path)
        handle = usd_scene.get_transform(camera_path)
        if not prim or handle is None or not prim.IsA(UsdGeom.Camera):
            continue
        camera = UsdGeom.Camera(prim)
        if str(camera.GetProjectionAttr().Get()) != "perspective":
            continue
        focal_length = float(camera.GetFocalLengthAttr().Get() or 0.0)
        aperture = float(camera.GetVerticalApertureAttr().Get() or 0.0)
        if focal_length <= 0.0 or aperture <= 0.0:
            continue
        world = usd_scene.get_world_transform(handle)
        position, yaw, pitch = _camera_angles(world[:3, 3], -world[:3, 2])
        fov = math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))
        return position, yaw, pitch, float(np.clip(fov, 5.0, 120.0)), camera_path
    if requested_path:
        raise ValueError(f"USD camera path is missing, invisible, or unsupported: {requested_path}")
    return None


def _camera_for_scene(api: PathTracer, requested_path=None, use_authored=True):
    bounds_min, bounds_max = _world_bounds(api)
    center = (bounds_min + bounds_max) * 0.5
    radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 1.0e-3)
    if use_authored:
        authored = _authored_camera(api, requested_path)
        if authored is not None:
            position, yaw, pitch, fov, camera_path = authored
            return position, yaw, pitch, fov, radius, camera_path
    # Favor a three-quarter view and keep the eye above the upper bound. Using
    # only a center-relative elevation can put the eye inside unusually flat
    # assets and expose their underside.
    position = center + radius * np.array((1.25, 0.65, 1.25), dtype=np.float32)
    position[1] = max(position[1], float(bounds_max[1]) + radius * 0.25)
    position, yaw, pitch = _camera_angles(position, center - position)
    return position, yaw, pitch, 55.0, radius, None


def _default_camera_speed(scene_radius: float) -> float:
    """Return a useful speed for centimeter- through world-scale stages."""
    return max(float(scene_radius) * 0.75, 0.25)


def main():
    args = _parse_args()
    if (args.camera_position is None) != (args.camera_target is None):
        raise ValueError("--camera-position and --camera-target must be specified together.")
    scene_usd = args.scene_usd.expanduser().resolve()
    if not scene_usd.is_file():
        raise FileNotFoundError(f"USD scene does not exist: {scene_usd}")

    wp.init()
    api = PathTracer(
        width=args.width,
        height=args.height,
        denoiser=args.denoiser,
        enable_cuda_graphs=not args.no_cuda_graphs,
        enable_set=not args.no_set,
        backface_culling=not args.no_backface_culling,
    )
    api.initialize()
    # Match Newton's ViewerOptix display defaults.
    api.tonemap_exposure = args.exposure
    api.tonemap_contrast = args.contrast
    api.tonemap_saturation = args.saturation
    material_debug_override = args.disable_normal_textures or args.disable_orm_textures
    if not api.load_scene_from_usd(
        str(scene_usd),
        build_scene=not material_debug_override,
        apply_stage_units=not args.no_stage_units,
        convert_up_axis=not args.no_up_axis_conversion,
        max_texture_size=args.max_texture_size or None,
        strict_sidedness=args.strict_usd_sidedness,
        load_usd_environment=args.load_usd_environment,
        usd_environment_scale=args.usd_environment_scale,
    ):
        raise RuntimeError(f"USD stage contained no supported render meshes: {scene_usd}")
    if material_debug_override:
        for material in api.scene.materials._materials:
            if args.disable_normal_textures:
                material["normalTexture"]["index"] = -1
            if args.disable_orm_textures:
                material["pbrMetallicRoughnessTexture"]["index"] = -1
                material["occlusionTexture"]["index"] = -1
                material["pbrMetallicFactor"] = 0.0
                material["pbrRoughnessFactor"] = 0.5
        api.scene.materials._dirty = True
        api.build_scene()
    api.set_debug_buffer_mode(args.debug_buffer_mode)

    cam_pos, cam_yaw, cam_pitch, cam_fov, scene_radius, camera_path = _camera_for_scene(
        api,
        requested_path=args.usd_camera,
        use_authored=not args.no_usd_camera,
    )
    if args.camera_position is not None:
        cam_pos, cam_yaw, cam_pitch = _camera_angles(
            args.camera_position,
            np.asarray(args.camera_target, dtype=np.float32) - np.asarray(args.camera_position, dtype=np.float32),
        )
        camera_path = None
    if args.camera_fov is not None:
        cam_fov = float(np.clip(args.camera_fov, 5.0, 120.0))
    # Bounds can be expressed in meters and legitimately be only a few
    # centimeters across. Keep the automatic speed useful for those stages
    # while retaining proportional movement for room- and world-scale scenes.
    camera_speed = args.camera_speed if args.camera_speed is not None else _default_camera_speed(scene_radius)
    render_width, render_height = int(args.width), int(args.height)
    last_elapsed = 0.0

    def _on_resize(width: int, height: int):
        nonlocal render_width, render_height, last_elapsed
        render_width, render_height = int(width), int(height)
        api.resize(render_width, render_height)
        last_elapsed = 0.0

    viewer = GLInteropViewer(
        width=args.width,
        height=args.height,
        device="cuda",
        title=args.title,
        fps=args.fps,
        on_resize=_on_resize,
        vsync=args.fps > 0,
    )
    controller = FreeCameraController(viewer, api, cam_pos, cam_yaw, cam_pitch, cam_fov, camera_speed)

    def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
        nonlocal last_elapsed
        controller.update(elapsed_sec - last_elapsed)
        last_elapsed = elapsed_sec
        api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(render_height, render_width),
            inputs=[api.viewer.tonemapped_output, mapped_image, render_width, render_height],
            device="cuda",
        )

    print(f"[optix] loaded USD scene: {scene_usd}")
    print(
        f"[optix] meshes={api.scene.mesh_count} materials={api.scene.materials.count} "
        f"textures={api.scene.texture_count}"
    )
    if args.camera_position is not None:
        print(
            f"[optix] camera=explicit position={tuple(args.camera_position)} "
            f"target={tuple(args.camera_target)} vertical_fov={cam_fov:.2f} degrees"
        )
    elif camera_path is not None:
        print(f"[optix] USD camera={camera_path} vertical_fov={cam_fov:.2f} degrees")
    else:
        print("[optix] camera=world-space geometry bounds fallback")
    if args.load_usd_environment and api.scene.usd_environment_path is not None:
        print(f"[optix] USD environment={api.scene.usd_environment_path} scale={args.usd_environment_scale:g}")
    viewer.run(_render, max_frames=args.max_frames)
    if args.screenshot is not None:
        from PIL import Image

        screenshot = args.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        frame = np.clip(api.get_frame(), 0.0, 1.0)
        Image.fromarray((frame[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(screenshot)
        print(f"[optix] saved screenshot: {screenshot}")


if __name__ == "__main__":
    main()
