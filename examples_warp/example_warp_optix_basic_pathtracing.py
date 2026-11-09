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

"""Pathtrace a procedural chess set or a user-supplied glTF/GLB scene."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import warp as wp
from warp_optix.gl import GLInteropViewer
from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    PathTracer,
)


@wp.kernel
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


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--scene-gltf",
        type=str,
        default=None,
        help="Optional path to a glTF/GLB scene. If omitted, generates a procedural chess scene.",
    )
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
    parser.add_argument("--title", type=str, default="Warp OptiX Basic Pathtracing")
    parser.add_argument("--camera-speed", type=float, default=2.0, help="Camera movement speed in scene units/second.")
    parser.add_argument("--camera-position", type=float, nargs=3, default=None)
    parser.add_argument("--camera-target", type=float, nargs=3, default=None)
    parser.add_argument("--camera-fov", type=float, default=None)
    parser.add_argument("--exposure", type=float, default=0.68, help="Linear display exposure multiplier.")
    parser.add_argument("--contrast", type=float, default=1.08, help="Display contrast multiplier.")
    parser.add_argument("--saturation", type=float, default=1.1, help="Display saturation multiplier.")
    parser.add_argument(
        "--denoiser",
        choices=("auto", "optix", "none"),
        default=None,
        help="Select the denoiser backend.",
    )
    parser.add_argument("--optix-upscale", action="store_true", help="Enable OptiX temporal 2x upscaling.")
    parser.add_argument("--no-set", action="store_true", help="Disable Shader Execution Reordering.")
    parser.add_argument(
        "--no-backface-culling",
        action="store_true",
        help="Render back-facing triangles for globally two-sided traversal.",
    )
    return parser.parse_args()


def _camera_angles(position, target):
    direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-20)
    yaw = math.degrees(math.atan2(float(direction[0]), float(direction[2])))
    pitch = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
    return tuple(position), yaw, pitch


def _resolve_scene_gltf(scene_gltf_arg: str | None) -> Path | None:
    if scene_gltf_arg is None:
        return None
    scene_gltf = Path(scene_gltf_arg).expanduser().resolve()
    if not scene_gltf.is_file():
        raise FileNotFoundError(f"--scene-gltf does not exist: {scene_gltf}")
    return scene_gltf


class FreeCameraController:
    """Newton-style interactive camera controls for the basic pathtracing example."""

    def __init__(self, viewer, api: PathTracer, position, yaw: float, pitch: float, fov: float, movement_speed: float):
        self.window = viewer.window
        self.pyglet = viewer.pyglet
        self.api = api
        self.position = np.array(position, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self._keys_down: set[int] = set()
        self._cam_speed = max(0.0, float(movement_speed))
        self._look_sensitivity = 0.1

        # Push initial camera state.
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)
        self.window.push_handlers(self)

    def _forward_right(self) -> tuple[np.ndarray, np.ndarray]:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cos_pitch = math.cos(pitch_rad)

        forward = np.array(
            [
                math.sin(yaw_rad) * cos_pitch,
                math.sin(pitch_rad),
                math.cos(yaw_rad) * cos_pitch,
            ],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1.0e-6:
            right /= right_norm
        return forward, right

    def update(self, dt: float):
        try:
            key = self.pyglet.window.key
        except Exception:
            # Fallback to no-op if backend doesn't expose key symbols.
            return

        forward, right = self._forward_right()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)

        if key.W in self._keys_down or key.UP in self._keys_down:
            move += forward
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            move -= forward
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            move -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            move += right
        if key.Q in self._keys_down:
            move -= world_up
        if key.E in self._keys_down:
            move += world_up

        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0e-6:
            speed = self._cam_speed
            if key.LSHIFT in self._keys_down or key.RSHIFT in self._keys_down:
                speed *= 4.0
            self.position += (move / move_norm) * speed * max(0.0, min(dt, 0.1))
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_key_press(self, symbol, _modifiers):
        try:
            key = self.pyglet.window.key
        except Exception:
            key = None

        if key is not None:
            mode_by_key = {
                key._1: 0,  # final
                key._2: 1,  # radiance
                key._3: 2,  # depth
                key._4: 3,  # motion
                key._5: 4,  # normal
                key._6: 5,  # roughness
                key._7: 6,  # diffuse
                key._8: 7,  # specular
                key._9: 8,  # spec hit distance
            }
            if symbol in mode_by_key:
                self.api.set_debug_buffer_mode(mode_by_key[symbol])

        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        try:
            mouse = self.pyglet.window.mouse
        except Exception:
            return

        if buttons & mouse.LEFT:
            self.yaw -= float(dx) * self._look_sensitivity
            self.pitch += float(dy) * self._look_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.fov = max(15.0, min(90.0, self.fov - float(scroll_y) * 2.0))
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)


def main():
    args = _parse_args()
    scene_gltf = _resolve_scene_gltf(args.scene_gltf)

    wp.init()

    api = PathTracer(
        width=args.width,
        height=args.height,
        denoiser=args.denoiser,
        optix_upscale=args.optix_upscale,
        enable_set=not args.no_set,
        backface_culling=not args.no_backface_culling,
    )
    api.initialize()
    # Match Newton's ViewerOptix display defaults.
    api.tonemap_exposure = args.exposure
    api.tonemap_contrast = args.contrast
    api.tonemap_saturation = args.saturation

    if scene_gltf is not None and not api.load_scene_from_gltf(str(scene_gltf), build_scene=True):
        raise RuntimeError(f"Failed to load glTF scene: {scene_gltf}")

    # Camera preset mirrored from Newton basic OptiX pathtracing example.
    cam_pos = (-0.803, 0.340, 0.327)
    cam_yaw = 115.2
    cam_pitch = -21.8
    cam_fov = 45.0
    if scene_gltf is None:
        from example_warp_optix_chess_scene import (
            CAMERA_FOV,
            CAMERA_POSITION,
            CAMERA_TARGET,
            build_chess_scene,
        )

        build_chess_scene(api)
        api.build_scene()
        cam_pos, cam_yaw, cam_pitch = _camera_angles(CAMERA_POSITION, CAMERA_TARGET)
        cam_fov = CAMERA_FOV
    if (args.camera_position is None) != (args.camera_target is None):
        raise ValueError("--camera-position and --camera-target must be supplied together")
    if args.camera_position is not None:
        cam_pos, cam_yaw, cam_pitch = _camera_angles(args.camera_position, args.camera_target)
    if args.camera_fov is not None:
        cam_fov = float(np.clip(args.camera_fov, 5.0, 120.0))

    render_width = int(args.width)
    render_height = int(args.height)
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

    # Attach Newton-style free camera controls to the viewer window.
    controller = FreeCameraController(viewer, api, cam_pos, cam_yaw, cam_pitch, cam_fov, args.camera_speed)

    def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
        nonlocal last_elapsed
        dt = elapsed_sec - last_elapsed
        last_elapsed = elapsed_sec
        controller.update(dt)
        api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(render_height, render_width),
            inputs=[api.viewer.tonemapped_output, mapped_image, render_width, render_height],
            device="cuda",
        )

    print(f"[optix] scene: {scene_gltf if scene_gltf is not None else 'procedural chess'}")
    print(
        f"[optix] denoiser: {api.active_denoiser}, "
        f"render: {api.viewer._render_width}x{api.viewer._render_height}, "
        f"output: {api.width}x{api.height}"
    )
    viewer.run(_render, max_frames=args.max_frames)
    if args.screenshot is not None:
        from PIL import Image

        screenshot = args.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        # The render buffer uses OpenGL's bottom-left origin.
        frame = np.flipud(np.clip(api.get_frame(), 0.0, 1.0))
        Image.fromarray((frame[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(screenshot)
        print(f"[optix] saved screenshot: {screenshot}")


if __name__ == "__main__":
    main()
