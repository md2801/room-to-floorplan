from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def extract_video_frames(video_path: str, out_dir: str, sample_every: int = 10, max_frames: int = 100) -> List[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths: List[str] = []
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            p = out / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(p), frame)
            paths.append(str(p))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return paths


def reconstruct_sparse_point_cloud(frame_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if len(frame_paths) < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    imgs = [cv2.imread(p) for p in frame_paths]
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    
    h, w = gray[0].shape[:2]
    focal = 0.9 * max(h, w)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

    orb = cv2.ORB_create(5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pts_3d = []
    cols = []
    prev_R = np.eye(3, dtype=np.float64)
    prev_t = np.zeros((3, 1), dtype=np.float64)

    for i in range(len(imgs) - 1):
        kp1, des1 = orb.detectAndCompute(gray[i], None)
        kp2, des2 = orb.detectAndCompute(gray[i + 1], None)
        
        if des1 is None or des2 is None:
            continue
            
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:1500]
        
        if len(matches) < 30:
            continue
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            continue
            
        _, R, t, mask_pose = cv2.recoverPose(E, p1, p2, K)
        mask_all = (mask.ravel() > 0) & (mask_pose.ravel() > 0)
        p1, p2 = p1[mask_all], p2[mask_all]
        
        if len(p1) < 20:
            continue

        P1 = K @ np.hstack([prev_R, prev_t])
        
        cur_R = R @ prev_R
        cur_t = prev_t + prev_R @ t
        
        P2 = K @ np.hstack([cur_R, cur_t])
        
        pts4 = cv2.triangulatePoints(P1, P2, p1.T, p2.T).T
        pts3 = pts4[:, :3] / np.maximum(pts4[:, 3:4], 1e-8)
        
        good = np.isfinite(pts3).all(axis=1) & (pts3[:, 2] > 0)
        pts3 = pts3[good]
        
        if len(pts3) == 0:
            continue
            
        p1v = p1[good].astype(int)
        p1v[:, 0] = np.clip(p1v[:, 0], 0, w - 1)
        p1v[:, 1] = np.clip(p1v[:, 1], 0, h - 1)
        
        c = imgs[i][p1v[:, 1], p1v[:, 0]][:, ::-1]
        pts_3d.append(pts3.astype(np.float32))
        cols.append(c.astype(np.uint8))
        prev_R, prev_t = cur_R, cur_t

    if not pts_3d:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        
    pts = np.concatenate(pts_3d, axis=0)
    colors = np.concatenate(cols, axis=0)

    # Normalize to make z up-ish by PCA fallback.
    pts = pts - pts.mean(axis=0, keepdims=True)
    
    return pts, colors
