# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-traced scene with 100k changing depth-aware OpenGL contact lines."""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp
from example_warp_optix_basic_pathtracing import (
    FreeCameraController,
    _pack_display_rgba8,
)
from example_warp_optix_pathtraced_arrowball import generate_arrow_ball
from example_warp_optix_pathtraced_scene import (
    RAINBOW_SRGB,
    add_checker_ground,
    configure_demo_sky,
)
from warp_optix.gl import GLInteropViewer, GLLineOverlay
from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    PathTracer,
)


@wp.kernel
def _animate_contacts(
    base_starts: wp.array(dtype=wp.vec3),
    base_ends: wp.array(dtype=wp.vec3),
    base_colors: wp.array(dtype=wp.vec3),
    phase: wp.float32,
    permutation_offset: wp.int32,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    line = wp.tid()
    count = base_starts.shape[0]
    # The odd multiplier is coprime to powers of two and produces a deliberately
    # chaotic source ordering as the offset changes every frame.
    source = (line * 8191 + permutation_offset) % count
    start = base_starts[source]
    direction = base_ends[source] - start
    pulse = 0.82 + 0.18 * wp.sin(phase + wp.float32(source % 97) * 0.071)
    starts[line] = start
    ends[line] = start + direction * pulse
    colors[line] = base_colors[source]


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--contact-count", type=int, default=100_000)
    parser.add_argument("--ball-radius", type=float, default=0.8)
    parser.add_argument("--line-length", type=float, default=0.32)
    parser.add_argument("--line-width", type=float, default=1.5)
    parser.add_argument("--line-antialias-width", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-depth-test", action="store_true")
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument("--denoiser", choices=("auto", "optix", "none"), default="auto")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--no-set", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.contact_count < 1:
        raise ValueError("contact_count must be positive")
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")

    host_starts, host_ends, slots = generate_arrow_ball(
        args.contact_count,
        args.ball_radius,
        args.line_length,
        swirl=0.14,
        seed=29,
    )
    host_colors = RAINBOW_SRGB[slots]

    wp.init()
    api = PathTracer(
        width=args.width,
        height=args.height,
        denoiser=args.denoiser,
        enable_cuda_graphs=not args.no_cuda_graphs,
        enable_set=not args.no_set,
    )
    api.initialize()
    api.tonemap_exposure = 0.32
    api.tonemap_contrast = 1.08
    api.tonemap_saturation = 1.1
    core_material = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    api.add_sphere((0.0, 0.0, 0.0), args.ball_radius * 1.005, 64, core_material)
    add_checker_ground(api, height=-1.38)
    api.build_scene()
    configure_demo_sky(api)

    base_starts = wp.array(host_starts, dtype=wp.vec3, device="cuda")
    base_ends = wp.array(host_ends, dtype=wp.vec3, device="cuda")
    base_colors = wp.array(host_colors, dtype=wp.vec3, device="cuda")
    dynamic_starts = wp.empty_like(base_starts)
    dynamic_ends = wp.empty_like(base_ends)
    dynamic_colors = wp.empty_like(base_colors)

    render_width, render_height = args.width, args.height
    last_elapsed = 0.0
    overlay = None

    def _on_resize(width: int, height: int):
        nonlocal render_width, render_height, last_elapsed
        render_width, render_height = int(width), int(height)
        api.resize(render_width, render_height)
        if overlay is not None:
            overlay.set_depth_buffer(api.linear_depth_output)
        last_elapsed = 0.0

    viewer = GLInteropViewer(
        width=args.width,
        height=args.height,
        device="cuda",
        title="Warp OptiX — Dynamic OpenGL Contact Lines",
        fps=args.fps,
        on_resize=_on_resize,
        vsync=args.fps > 0,
    )
    overlay = GLLineOverlay(
        viewer.gl,
        args.contact_count,
        device="cuda",
        depth_buffer=api.linear_depth_output,
        line_width=args.line_width,
        antialias_width=args.line_antialias_width,
        stream=viewer.render_stream,
        use_depth_test=not args.no_depth_test,
    )

    draw_seconds = 0.0
    drawn_frames = 0

    def _draw_overlay():
        nonlocal draw_seconds, drawn_frames
        camera = api.viewer.camera
        draw_start = time.perf_counter()
        overlay.draw(
            camera.get_view_matrix(),
            camera.get_projection_matrix(),
            (viewer.width, viewer.height),
            camera_near=camera.near,
            camera_far=camera.far,
        )
        draw_seconds += time.perf_counter() - draw_start
        drawn_frames += 1

    viewer.set_draw_overlay(_draw_overlay)
    controller = FreeCameraController(viewer, api, (0.0, 0.18, 3.55), 180.0, -3.0, 42.0, args.camera_speed)
    update_seconds = 0.0
    measured_frames = 0

    def _render(mapped_image: wp.array, frame: int, elapsed_sec: float):
        nonlocal last_elapsed, update_seconds, measured_frames
        controller.update(elapsed_sec - last_elapsed)
        last_elapsed = elapsed_sec
        # Exercise changing counts as well as a fully shuffled order.
        active_count = max(
            1,
            int(args.contact_count * (0.75 + 0.25 * (0.5 + 0.5 * np.sin(elapsed_sec)))),
        )
        update_start = time.perf_counter()
        wp.launch(
            _animate_contacts,
            dim=active_count,
            inputs=[
                base_starts,
                base_ends,
                base_colors,
                elapsed_sec * 2.0,
                (frame * 8191) % args.contact_count,
            ],
            outputs=[dynamic_starts, dynamic_ends, dynamic_colors],
            device="cuda",
        )
        overlay.update_device(dynamic_starts, dynamic_ends, dynamic_colors, count=active_count)
        update_seconds += time.perf_counter() - update_start
        measured_frames += 1
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

    print(
        f"[opengl] dynamic contact overlay: capacity {args.contact_count:,} lines; count and order change every frame"
    )
    print("[opengl] controls: left-drag look, WASD move, Q/E down/up, wheel zoom")
    try:
        viewer.run(_render, max_frames=args.max_frames)
    finally:
        if measured_frames:
            print(
                f"[opengl] average contact generation + VBO update: {1000.0 * update_seconds / measured_frames:.3f} ms"
            )
        if drawn_frames:
            print(f"[opengl] average depth upload + GL draw: {1000.0 * draw_seconds / drawn_frames:.3f} ms")
        overlay.close()


if __name__ == "__main__":
    main()
