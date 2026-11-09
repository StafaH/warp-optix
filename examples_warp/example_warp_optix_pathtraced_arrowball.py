# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive rainbow arrow ball using one native-curve ArrowBatch."""

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


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arrow-count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ball-radius", type=float, default=0.8)
    parser.add_argument("--arrow-length", type=float, default=0.42)
    parser.add_argument("--swirl", type=float, default=0.24, help="Tangential direction bias.")
    parser.add_argument("--small-radius", type=float, default=0.011)
    parser.add_argument("--large-radius", type=float, default=0.034)
    parser.add_argument("--tip-length-ratio", type=float, default=0.28)
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
    parser.add_argument("--title", default="Warp OptiX Path-traced Arrow Ball")
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=0.32)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--saturation", type=float, default=1.1)
    parser.add_argument("--denoiser", choices=("auto", "optix", "none"), default="auto")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--no-set", action="store_true")
    return parser.parse_args()


def generate_arrow_ball(
    arrow_count: int,
    ball_radius: float,
    arrow_length: float,
    swirl: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate randomized outward arrows on a Fibonacci sphere."""
    if arrow_count < 1:
        raise ValueError("arrow_count must be positive")
    if ball_radius <= 0.0 or arrow_length <= 0.0:
        raise ValueError("ball_radius and arrow_length must be positive")
    if swirl < 0.0:
        raise ValueError("swirl must be non-negative")

    rng = np.random.default_rng(seed)
    arrow_id = np.arange(arrow_count, dtype=np.float64)
    y = 1.0 - 2.0 * (arrow_id + 0.5) / arrow_count
    azimuth = arrow_id * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.cos(azimuth),
            y,
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.sin(azimuth),
        )
    )
    radial += rng.normal(scale=0.008, size=radial.shape)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    reference = np.zeros_like(radial)
    reference[:, 1] = 1.0
    reference[np.abs(radial[:, 1]) > 0.9] = (1.0, 0.0, 0.0)
    tangent = np.cross(reference, radial)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(radial, tangent)
    phase = rng.uniform(0.0, 2.0 * np.pi, (arrow_count, 1))
    tangential = np.cos(phase) * tangent + np.sin(phase) * bitangent
    direction = radial + float(swirl) * tangential
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    starts = float(ball_radius) * radial
    lengths = float(arrow_length) * rng.uniform(0.78, 1.22, (arrow_count, 1))
    ends = starts + lengths * direction
    slots = rainbow_height_slots(starts)
    return starts.astype(np.float32), ends.astype(np.float32), slots


def main():
    args = _parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")
    starts, ends, palette_slots = generate_arrow_ball(
        args.arrow_count,
        args.ball_radius,
        args.arrow_length,
        args.swirl,
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

    core_material = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    palette_materials = create_rainbow_materials(api)
    api.add_sphere((0.0, 0.0, 0.0), args.ball_radius * 1.005, 64, core_material)
    add_checker_ground(api, height=-1.38)
    arrows = api.create_arrow_batch(
        capacity=args.arrow_count,
        small_radius=args.small_radius,
        large_radius=args.large_radius,
        tip_length_ratio=args.tip_length_ratio,
        material_id=int(palette_materials[0]),
        material_ids=palette_materials[palette_slots],
    )
    api.update_arrow_batch(arrows, starts, ends)
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

    print(f"[optix] arrow ball: {args.arrow_count:,} arrows, {2 * args.arrow_count:,} curve primitives")
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
