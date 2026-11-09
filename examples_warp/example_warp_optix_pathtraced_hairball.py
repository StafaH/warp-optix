# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive PBR path-traced hair ball using native OptiX curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from example_warp_optix_basic_pathtracing import (
    FreeCameraController,
    _pack_display_rgba8,
)
from example_warp_optix_pathtraced_scene import (
    add_checker_ground,
    configure_demo_sky,
    create_rainbow_materials,
    rainbow_height_slots,
)
from warp_optix.gl import GLInteropViewer
from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    PathTracer,
)


def rainbow_segment_slots(points: np.ndarray, segments: int) -> np.ndarray:
    """Assign one palette slot per segment using gently rippled height bands."""
    roots = points[:: 3 * segments + 1]
    strand_slots = rainbow_height_slots(roots)
    return np.repeat(strand_slots, segments)


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hair-count", type=int, default=4000)
    parser.add_argument("--segments", type=int, default=5, help="Cubic Bezier segments per hair.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ball-radius", type=float, default=0.8)
    parser.add_argument("--hair-length", type=float, default=0.48)
    parser.add_argument("--curl-radius", type=float, default=0.065)
    parser.add_argument("--curl-turns", type=float, default=1.65)
    parser.add_argument("--root-radius", type=float, default=0.012)
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--fps", type=int, default=0, help="Presentation rate cap; 0 is unlimited.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Auto-exit after N frames; 0 runs forever.",
    )
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--title", default="Warp OptiX Path-traced Hair Ball")
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=0.32)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--saturation", type=float, default=1.1)
    parser.add_argument("--denoiser", choices=("auto", "optix", "none"), default="auto")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--no-set", action="store_true")
    return parser.parse_args()


def generate_hair_ball(
    hair_count: int,
    segments: int,
    ball_radius: float,
    hair_length: float,
    curl_radius: float,
    curl_turns: float,
    root_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack disjoint tapered helical hairs as C1-continuous cubic Beziers."""
    if hair_count < 1:
        raise ValueError("hair_count must be positive")
    if segments < 2:
        raise ValueError("segments must be at least 2")
    if min(ball_radius, hair_length, root_radius) <= 0.0 or curl_radius < 0.0:
        raise ValueError("ball_radius, hair_length, and root_radius must be positive")

    rng = np.random.default_rng(seed)
    root_id = np.arange(hair_count, dtype=np.float64)
    y = 1.0 - 2.0 * (root_id + 0.5) / hair_count
    azimuth = root_id * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.cos(azimuth),
            y,
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.sin(azimuth),
        )
    )
    radial += rng.normal(scale=0.012, size=radial.shape)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    reference = np.zeros_like(radial)
    reference[:, 1] = 1.0
    reference[np.abs(radial[:, 1]) > 0.9] = (1.0, 0.0, 0.0)
    tangent = np.cross(reference, radial)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(radial, tangent)

    phase = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    length = hair_length * rng.uniform(0.78, 1.18, (hair_count, 1))
    turns = curl_turns * rng.uniform(0.72, 1.28, (hair_count, 1))
    curl = curl_radius * rng.uniform(0.65, 1.35, (hair_count, 1))
    lean_angle = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    lean = np.cos(lean_angle)[..., None] * tangent[:, None, :] + np.sin(lean_angle)[..., None] * bitangent[:, None, :]

    u = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)[None, :]
    angle = phase + 2.0 * np.pi * turns * u
    envelope = np.sin(0.5 * np.pi * u) ** 1.25
    helix = np.cos(angle)[..., None] * tangent[:, None, :] + np.sin(angle)[..., None] * bitangent[:, None, :]
    center = (ball_radius + length * u)[..., None] * radial[:, None, :]
    knots = center + (curl * envelope)[..., None] * helix + (0.13 * length * u * u)[..., None] * lean

    width_scale = rng.uniform(0.8, 1.2, (hair_count, 1))
    radius_knots = root_radius * width_scale * (0.08 + 0.92 * (1.0 - u) ** 0.75)
    step = 1.0 / segments
    derivatives = np.empty_like(knots)
    derivatives[:, 0] = (knots[:, 1] - knots[:, 0]) / step
    derivatives[:, -1] = (knots[:, -1] - knots[:, -2]) / step
    derivatives[:, 1:-1] = (knots[:, 2:] - knots[:, :-2]) / (2.0 * step)
    radius_derivatives = np.empty_like(radius_knots)
    radius_derivatives[:, 0] = (radius_knots[:, 1] - radius_knots[:, 0]) / step
    radius_derivatives[:, -1] = (radius_knots[:, -1] - radius_knots[:, -2]) / step
    radius_derivatives[:, 1:-1] = (radius_knots[:, 2:] - radius_knots[:, :-2]) / (2.0 * step)

    points_per_hair = 3 * segments + 1
    points = np.empty((hair_count, points_per_hair, 3), dtype=np.float64)
    radii = np.empty((hair_count, points_per_hair), dtype=np.float64)
    points[:, 0] = knots[:, 0]
    radii[:, 0] = radius_knots[:, 0]
    points[:, 1::3] = knots[:, :-1] + derivatives[:, :-1] * (step / 3.0)
    points[:, 2::3] = knots[:, 1:] - derivatives[:, 1:] * (step / 3.0)
    points[:, 3::3] = knots[:, 1:]
    radii[:, 1::3] = radius_knots[:, :-1] + radius_derivatives[:, :-1] * (step / 3.0)
    radii[:, 2::3] = radius_knots[:, 1:] - radius_derivatives[:, 1:] * (step / 3.0)
    radii[:, 3::3] = radius_knots[:, 1:]
    radii = np.maximum(radii, root_radius * 0.01)
    segment_indices = (
        np.arange(hair_count, dtype=np.uint32)[:, None] * points_per_hair
        + 3 * np.arange(segments, dtype=np.uint32)[None, :]
    ).reshape(-1)
    return (
        points.astype(np.float32).reshape(-1, 3),
        radii.astype(np.float32).reshape(-1),
        segment_indices,
    )


def main():
    args = _parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")
    points, radii, segment_indices = generate_hair_ball(
        args.hair_count,
        args.segments,
        args.ball_radius,
        args.hair_length,
        args.curl_radius,
        args.curl_turns,
        args.root_radius,
        args.seed,
    )

    wp.init()
    api = PathTracer(
        width=args.width,
        height=args.height,
        denoiser=args.denoiser,
        enable_cuda_graphs=not args.no_cuda_graphs,
        enable_set=not args.no_set,
    )
    api.initialize()
    api.tonemap_exposure = args.exposure
    api.tonemap_contrast = args.contrast
    api.tonemap_saturation = args.saturation

    core_mat = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    palette_materials = create_rainbow_materials(api)
    segment_material_ids = palette_materials[rainbow_segment_slots(points, args.segments)]
    api.add_sphere((0.0, 0.0, 0.0), args.ball_radius * 1.005, 64, core_mat)
    add_checker_ground(api, height=-1.38)
    curve_id = api.create_curve(
        points,
        radii,
        segment_indices,
        material_id=int(palette_materials[0]),
        material_ids=segment_material_ids,
        basis="cubic_bezier",
    )
    api.create_instance(curve_id)
    api.build_scene()
    configure_demo_sky(api)

    render_width, render_height = args.width, args.height
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
    controller = FreeCameraController(viewer, api, (0.0, 0.18, 3.55), 180.0, -3.0, 42.0, args.camera_speed)

    def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
        nonlocal last_elapsed
        controller.update(elapsed_sec - last_elapsed)
        last_elapsed = elapsed_sec
        api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(render_height, render_width),
            inputs=[
                api.viewer.tonemapped_output,
                mapped_image,
                render_width,
                render_height,
            ],
            device="cuda",
        )

    print(f"[optix] hair ball: {args.hair_count:,} strands, {len(segment_indices):,} cubic Bezier segments")
    print("[optix] controls: left-drag look, WASD move, Q/E down/up, wheel zoom")
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
