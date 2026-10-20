"""Templates (apply the settings of one project to another) and the batch builder."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from datetime import datetime

from .imageio import is_image_file, read_exif
from .project import Project


@dataclass
class TemplateBehavior:
    """What a template copies (mirrors the Project Settings -> Template behavior options)."""

    lens_and_crop: bool = True
    image_parameters: bool = True
    masks: bool = True
    panorama_settings: bool = True
    optimizer_settings: bool = True
    link_images: str = "as_template"  # as_template | auto_exif | keep
    camera_curve: str = "yes"  # yes | no | non_raw
    preserve_orientation: bool = False


def apply_template(target: Project, template: Project, behavior: TemplateBehavior | None = None) -> Project:
    """Copy geometry/settings from ``template`` onto ``target`` (images matched by index)."""
    # the template project carries its own behaviour settings (Project Settings tab)
    b = behavior or TemplateBehavior(**template.metadata.get("template_behavior", {}))
    n = min(len(target.images), len(template.images))
    if b.lens_and_crop:
        target.lenses = copy.deepcopy(template.lenses)
        for i in range(len(target.images)):
            src = template.images[min(i, len(template.images) - 1)]
            target.images[i].lens = min(src.lens, len(target.lenses) - 1)
            target.images[i].lens_overrides = dict(src.lens_overrides)
        if b.camera_curve == "no":
            for ln in target.lenses:
                ln.response, ln.response_params = "srgb", [2.2]
    if b.image_parameters:
        for i in range(n):
            s, t = template.images[i], target.images[i]
            for k in ("yaw", "pitch", "roll", "include", "vp_enabled", "vp_x", "vp_y", "vp_z", "vp_pan", "vp_tilt",
                      "ev", "wb_r", "wb_b", "flare", "blend_priority"):
                setattr(t, k, getattr(s, k))
    if b.link_images == "as_template":
        for i in range(n):
            target.images[i].link_group = template.images[i].link_group
    elif b.link_images == "auto_exif":
        from .align import link_brackets

        for im in target.images:
            im.link_group = -1
        link_brackets(target)
    if b.masks:
        for i in range(n):
            if template.images[i].mask:
                target.images[i].mask = template.abspath(template.images[i].mask)
    if b.panorama_settings:
        target.panorama = copy.deepcopy(template.panorama)
        target.exposure = copy.deepcopy(template.exposure)
        out = copy.deepcopy(template.output)
        out.path = target.output.path
        target.output = out
    if b.optimizer_settings:
        target.optimizer = copy.deepcopy(template.optimizer)
    return target


# --------------------------------------------------------------------------- batch builder


@dataclass
class BatchGroup:
    images: list = field(default_factory=list)
    start: datetime | None = None


def _shot_time(path: str):
    ex = read_exif(path)
    dt = ex.get("datetime")
    if dt:
        try:
            return datetime.strptime(dt[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def group_by_time(paths, max_gap_s: float = 10.0, images_per_pano: int | None = None) -> list[BatchGroup]:
    """Split a sequence of shots into panoramas by capture-time gaps (or a fixed count)."""
    items = sorted(((_shot_time(p), p) for p in paths), key=lambda x: (x[0], x[1]))
    groups: list[BatchGroup] = []
    if images_per_pano:
        for k in range(0, len(items), images_per_pano):
            chunk = items[k:k + images_per_pano]
            groups.append(BatchGroup([p for _, p in chunk], chunk[0][0]))
        return groups
    cur = None
    last = None
    for t, p in items:
        if cur is None or (t - last).total_seconds() > max_gap_s:
            cur = BatchGroup([], t)
            groups.append(cur)
        cur.images.append(p)
        last = t
    return groups


def build_batch(folder_or_paths, out_dir: str, template: Project | None = None, max_gap_s: float = 10.0,
                images_per_pano: int | None = None, min_images: int = 2) -> list[str]:
    """Create one project per detected panorama.  Returns the project paths."""
    from .align import create_project

    if isinstance(folder_or_paths, str) and os.path.isdir(folder_or_paths):
        paths = sorted(os.path.join(folder_or_paths, f) for f in os.listdir(folder_or_paths) if is_image_file(f))
    else:
        paths = list(folder_or_paths)
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for g in group_by_time(paths, max_gap_s, images_per_pano):
        if len(g.images) < min_images:
            continue
        p = create_project(g.images)
        if template is not None:
            apply_template(p, template)
        name = os.path.splitext(os.path.basename(g.images[0]))[0] + " Panorama.pfproj"
        path = os.path.join(out_dir, name)
        p.save(path)
        out.append(path)
    return out
