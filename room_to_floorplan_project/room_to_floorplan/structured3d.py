from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PerspectiveView:
    room_id: str
    position_id: str
    rgb: np.ndarray
    depth_mm: np.ndarray
    normal: np.ndarray
    semantic: np.ndarray
    layout: dict
    pose: np.ndarray  # 4x4 cam-to-world
    intrinsics: np.ndarray  # 3x3


@dataclass
class PanoramaView:
    room_id: str
    rgb: np.ndarray
    depth_mm: np.ndarray
    normal: np.ndarray
    semantic: np.ndarray
    layout: np.ndarray
    camera_xyz_mm: np.ndarray


STRUCTURE_IDS = {
    1,  # wall
    2,  # floor
    22,  # ceiling
    8,  # door
    9   # window
}

FURNITURE_IDS = {
    5,  # chair
    6,  # sofa
    10,  # table
    11,  # desk
    12,  # bed
    14,  # wardrobe
    15,  # cabinet
    16,  # shelf
    17,  # dresser
    18,  # fridge
    19,  # sink
    24,  # bathtub
    25,  # toilet
    28,  # tv
    29,  # microwave
    30,  # oven
    31,  # dishwasher
    32,  # washing machine
    33,  # dryer
    34  # other furniture
}


class Structured3DZip:
    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        self.zf = zipfile.ZipFile(self.zip_path)
        self.names = set(self.zf.namelist())

    def _read_image(self, name: str, flags: Optional[int] = None) -> np.ndarray:
        data = self.zf.read(name)
        arr = np.frombuffer(data, dtype=np.uint8)
        if flags is None:
            img = Image.open(io.BytesIO(data))
            return np.array(img)
        out = cv2.imdecode(arr, flags)
        if out is None:
            raise FileNotFoundError(name)
        return out

    def _read_text(self, name: str) -> str:
        return self.zf.read(name).decode("utf-8")

    def _scene_root(self) -> str:
        for n in self.names:
            if "/scene_00000/2D_rendering/" in n:
                return n.split("scene_00000/2D_rendering/")[0] + "scene_00000/2D_rendering"
        raise FileNotFoundError("Could not find scene_00000/2D_rendering inside zip")

    def has_panorama(self) -> bool:
        return any("/panorama/" in n for n in self.names)

    def has_perspective(self) -> bool:
        return any("/perspective/" in n for n in self.names)

    def list_rooms(self) -> List[str]:
        rooms = []
        prefix = self._scene_root() + "/"
        for n in self.names:
            if n.startswith(prefix):
                rel = n[len(prefix):]
                parts = rel.split("/")
                if len(parts) > 1 and parts[0].isdigit():
                    rooms.append(parts[0])
        return sorted(set(rooms))

    def iter_panorama_views(self) -> Iterable[PanoramaView]:
        root = self._scene_root()
        for room_id in self.list_rooms():
            base = f"{root}/{room_id}/panorama"
            if f"{base}/full/rgb_rawlight.png" not in self.names:
                continue

            rgb = self._read_image(f"{base}/full/rgb_rawlight.png")
            depth_mm = self._read_image(f"{base}/full/depth.png")
            normal = self._read_image(f"{base}/full/normal.png")
            semantic = self._read_image(f"{base}/full/semantic.png")
            layout_txt = self._read_text(f"{base}/layout.txt")
            layout = np.loadtxt(io.StringIO(layout_txt), dtype=np.float32)
            camera_xyz_mm = np.loadtxt(
                io.StringIO(self._read_text(f"{base}/camera_xyz.txt")),
                dtype=np.float32
            )

            yield PanoramaView(
                room_id=room_id,
                rgb=rgb,
                depth_mm=depth_mm,
                normal=normal,
                semantic=semantic,
                layout=layout,
                camera_xyz_mm=camera_xyz_mm,
            )

    def iter_perspective_views(self) -> Iterable[PerspectiveView]:
        root = self._scene_root()
        for room_id in self.list_rooms():
            base = f"{root}/{room_id}/perspective/full"

            positions = sorted({
                n.split("/")[-2]
                for n in self.names
                if n.startswith(base + "/") and n.endswith("/rgb_rawlight.png")
            })

            for pid in positions:
                pbase = f"{base}/{pid}"

                rgb = self._read_image(f"{pbase}/rgb_rawlight.png")
                depth_mm = self._read_image(f"{pbase}/depth.png")
                normal = self._read_image(f"{pbase}/normal.png")
                semantic = self._read_image(f"{pbase}/semantic.png")
                layout = json.loads(self._read_text(f"{pbase}/layout.json"))

                pose_line = np.loadtxt(
                    io.StringIO(self._read_text(f"{pbase}/camera_pose.txt")),
                    dtype=np.float32
                ).reshape(-1)

                pose, intr = parse_perspective_pose_line(
                    pose_line, rgb.shape[1], rgb.shape[0]
                )

                yield PerspectiveView(
                    room_id=room_id,
                    position_id=pid,
                    rgb=rgb,
                    depth_mm=depth_mm,
                    normal=normal,
                    semantic=semantic,
                    layout=layout,
                    pose=pose,
                    intrinsics=intr,
                )


def parse_perspective_pose_line(line: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    # vx vy vz tx ty tz ux uy uz xfov yfov 1
    vx, vy, vz, tx, ty, tz, ux, uy, uz, xfov, yfov, _ = line.tolist()
    eye = np.array([vx, vy, vz], dtype=np.float32) / 1000.0
    target = np.array([tx, ty, tz], dtype=np.float32) / 1000.0
    up = np.array([ux, uy, uz], dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    # camera coordinates: x right, y down, z forward
    R = np.stack([right, -true_up, forward], axis=1)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = eye

    fx = (width / 2.0) / math.tan(float(xfov))
    fy = (height / 2.0) / math.tan(float(yfov))
    cx, cy = width / 2.0, height / 2.0
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return pose, intr


def decode_normals(normal_img: np.ndarray) -> np.ndarray:
    n = normal_img.astype(np.float32) / 128.0 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-6)
