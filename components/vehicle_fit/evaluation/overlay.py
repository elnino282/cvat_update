"""Camera projection and cloud/mask/box evidence, not repaired annotations."""
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .metrics import project


def edges():
    return [(a, b) for a, b in combinations(range(8), 2) if (a ^ b) in (1, 2, 4)]


def overlays(path, image_path, cloud, gt, prediction, mask_indices, seed, transform, p2, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(camera)
    draw.rectangle((0, 0, camera.width, 34), fill="black")
    draw.text((8, 8), title, fill="white")
    mask_indices = np.asarray(mask_indices, dtype=int)
    mask_indices = mask_indices[(mask_indices >= 0) & (mask_indices < len(cloud))]
    if len(mask_indices):
        projected, valid = project(cloud[mask_indices, :3], transform, p2)
        for x, y in projected[valid][::max(1, len(projected) // 3000)]:
            if 0 <= x < camera.width and 0 <= y < camera.height:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="cyan")
    for box, color in ((gt, "lime"), (prediction, "red")):
        if box is None:
            continue
        pixels, valid = project(box.corners, transform, p2)
        for a, b in edges():
            if valid[a] and valid[b]:
                draw.line((*pixels[a], *pixels[b]), fill=color, width=2)
    seed_pixel, valid = project(np.asarray([seed]), transform, p2)
    if valid[0]:
        x, y = seed_pixel[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="yellow")
    camera.save(path.with_name(path.name + "-camera.png"))
    bev = Image.new("RGB", (800, 640), "#161b22")
    draw = ImageDraw.Draw(bev)
    center = gt.center[:2]
    def screen(points):
        points = np.asarray(points)
        return np.column_stack((400 - (points[:, 1] - center[1]) * 35,
                                340 - (points[:, 0] - center[0]) * 35))
    local = cloud[np.all(np.abs(cloud[:, :2] - center) < np.array([9, 11]), axis=1)]
    for x, y in screen(local[::max(1, len(local) // 15000)]):
        draw.point((x, y), fill="#687483")
    for x, y in screen(cloud[mask_indices]):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="cyan")
    for box, color in ((gt, "lime"), (prediction, "red")):
        if box is not None:
            pixels = screen(box.corners)
            for a, b in edges():
                draw.line((*pixels[a], *pixels[b]), fill=color, width=2)
    x, y = screen(np.asarray([seed]))[0]
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="yellow")
    draw.text((10, 10), title, fill="white")
    draw.text((10, 30), "green: GT, red: inferred, cyan: instance mask, yellow: observed seed", fill="white")
    bev.save(path.with_name(path.name + "-cloud.png"))
