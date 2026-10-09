#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 PBR 反照率贴图里"金属全黑"的区域提亮，好让它在非 PBR 的游戏引擎里也能看见。

为什么需要：PBR 工作流里金属的反照率（base color）本来就接近纯黑，颜色信息放在
reflectivity / metalness 通道里。但很多老引擎只认一张 diffuse 贴图，于是金属部分
直接渲染成死黑。这个脚本把那些区域"抬"成亮色，算是一种近似补偿。

用法::

    :: 先看一眼效果（生成左右对比图，不改动任何贴图）
    python fix_metal_albedo.py 解包目录 --preview

    :: 直接修（生成 mat0_c_fixed.jpg，原图保留）
    python fix_metal_albedo.py 解包目录

    :: 只用反射率贴图当金属遮罩，并指定填充色
    python fix_metal_albedo.py 解包目录 --strategy reflectivity --fill 1,0.85,0.4

两种识别策略（默认 threshold）：

  * ``threshold``    —— 反照率亮度低于阈值的像素就当成"太黑"，直接提亮。
    简单、可预测，任何贴图都能用；缺点是分不清"金属黑"和"本来就是深色的布料/暗部"。

  * ``reflectivity`` —— 用 reflectivityTex（_r 贴图）当金属遮罩，只动金属。
    理论上更准，但**实测只有部分模型成立**：GCrestShield 的 _r 是干净的双峰分布
    （金属区反照率均值 0.022，非金属 0.246，区分得很开）；而 Sparda 的 mat1_r
    金属区反而更亮（0.563 vs 0.304），Trident / toolbag 的 _r 几乎没有区分度。
    Otsu 自动阈值在各材质间从 0.195 漂到 0.461，没有一个通用值。
    所以这个策略请配合 ``--dry-run`` 先看看遮罩覆盖率是否合理。

提亮方式不是"整片涂白"，而是把原图的明暗细节映射到 [floor, 1] 区间再染色，
这样划痕、磨损之类的细节会保留下来，比纯色填充好看很多：

    输出 = 填充色 × (floor + (1 - floor) × clamp(原亮度 / detail_max, 0, 1))

依赖 numpy 和 Pillow。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def luminance(rgb):
    return rgb @ LUMA


def otsu(values):
    """大津法自动阈值，返回 (阈值, 分离度)。分离度越小说明越不适合当遮罩。"""
    hist, edges = np.histogram(values, bins=128, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = values.size
    if total == 0:
        return 0.5, 0.0
    best_t, best_v = 0.5, 0.0
    for i in range(1, 127):
        n0 = hist[:i].sum()
        n1 = hist[i:].sum()
        if n0 == 0 or n1 == 0:
            continue
        w0 = n0 / total
        w1 = n1 / total
        mu0 = (hist[:i] * centers[:i]).sum() / n0
        mu1 = (hist[i:] * centers[i:]).sum() / n1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_v:
            best_v, best_t = var, centers[i]
    # 两类的均值差就是"分离度"，越大越说明确实是双峰
    return best_t, float(np.sqrt(best_v) * 2.0)


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array, path, quality=95):
    data = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(data, "RGB")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        # subsampling=0 -> 4:4:4，避免色度二次采样把贴图细节糊掉
        img.save(path, quality=quality, subsampling=0)
    else:
        img.save(path)
    return path


def box_blur_mask(mask, radius):
    """给遮罩做一点羽化，避免金属/非金属边界出现硬边。"""
    if radius <= 0:
        return mask
    img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------
# 核心变换
# --------------------------------------------------------------------------

def soft_below(values, lo, hi):
    """软阈值：<= lo 返回 1，>= hi 返回 0，中间平滑过渡。

    用硬阈值（非黑即白）在噪声多的贴图上会产生大量椒盐斑点：相邻像素一个刚好在
    阈值下方、一个刚好在上方，提亮后明暗交替，看起来像撒了盐。实测 FumeUltra 那把
    烟熏大剑就是这种贴图，硬阈值下整张图像是雪花点。改成斜坡过渡后斑点就没了。

    ``lo`` 会被夹到 0：否则（比如 threshold=0.06 / soft=0.15 时 lo=-0.09）
    纯黑像素的遮罩值只有 0.7，永远拿不到完整提亮，调参数时会很反直觉。
    """
    lo = max(0.0, lo)
    if hi <= lo:
        return (values <= lo).astype(np.float32)
    return np.clip((hi - values) / (hi - lo), 0.0, 1.0)


def soft_above(values, lo, hi):
    """软阈值：>= hi 返回 1，<= lo 返回 0，中间平滑过渡。"""
    lo = max(0.0, lo)
    if hi <= lo:
        return (values >= hi).astype(np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def saturation_of(rgb):
    """每像素的饱和度（0 = 中性灰，1 = 纯色）。"""
    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    return (hi - lo) / np.maximum(hi, 1e-6)


def build_mask(albedo, reflectivity, options, stats):
    """返回 0~1 的遮罩：越接近 1 表示"这里该被提亮"。

    遮罩用**模糊后的**亮度来算，这一点很关键：如果逐像素判断"这个像素够不够黑"，
    噪声贴图上相邻像素会一个进遮罩一个不进，结果就是一片椒盐斑点。
    区域归属本来就应该看局部平均值，而不是单像素。
    """
    soft = max(0.0, options.soft)

    if options.strategy == "reflectivity":
        if reflectivity is None:
            stats["note"] = "没有 reflectivity 贴图，退回 threshold 策略"
            options.strategy_used = "threshold"
        else:
            r = smooth_luminance(luminance(reflectivity), options.detail_blur)
            if options.reflectivity_threshold == "auto":
                threshold, separation = otsu(r)
                stats["otsu"] = float(threshold)
                stats["separation"] = separation
                if separation < 0.10:
                    stats["note"] = ("_r 分布接近单峰（分离度 %.3f），遮罩可能不可靠，"
                                     "建议改用 threshold 策略" % separation)
                mask = soft_above(r, threshold - soft, threshold + soft)
            else:
                t = float(options.reflectivity_threshold)
                mask = soft_above(r, t - soft, t + soft)
        return apply_saturation_guard(mask, albedo, options)

    lum = luminance(albedo)
    local = smooth_luminance(lum, options.detail_blur)
    mask = soft_below(local, options.threshold - soft, options.threshold + soft)
    return apply_saturation_guard(mask, albedo, options)


def apply_saturation_guard(mask, albedo, options):
    """有颜色的像素少提亮一些，保护彩色贴花（比如盾牌上的金色/青色花纹）。

    花纹的亮度往往也很低，只按亮度判断会被一起提亮、把图案洗掉。
    按饱和度打个折扣就能保住它们。
    """
    protect = getattr(options, "saturation_protect", 0.0)
    if protect <= 0:
        return mask
    sat = saturation_of(albedo)
    return mask * np.clip(1.0 - protect * sat, 0.0, 1.0)


def _box_blur_axis(array, radius, axis):
    """沿某一轴做滑动窗口均值（用前缀和，O(n)）。"""
    if radius < 1:
        return array
    n = array.shape[axis]
    window = 2 * radius + 1
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="reflect")

    # 用 float64 累积，避免 2048 个 float32 相加的精度损失
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zeros_shape = list(cumulative.shape)
    zeros_shape[axis] = 1
    cumulative = np.concatenate(
        [np.zeros(zeros_shape, dtype=np.float64), cumulative], axis=axis)

    high = [slice(None)] * array.ndim
    high[axis] = slice(window, window + n)
    low = [slice(None)] * array.ndim
    low[axis] = slice(0, n)
    return ((cumulative[tuple(high)] - cumulative[tuple(low)]) / float(window)).astype(np.float32)


def smooth_luminance(lum, radius):
    """低频版本的亮度图（两次可分离盒式模糊，近似高斯）。

    不用 PIL 的 GaussianBlur：它不支持 float 模式（'F'），
    而转成 8 位再模糊会把这个模型里 0.0075 这种很暗的亮度直接量化成 2/255，
    低频信息就废了。
    """
    if radius < 1:
        return lum
    r = max(1, int(round(radius / 2.0)))
    result = lum
    for _ in range(2):
        result = _box_blur_axis(result, r, 0)
        result = _box_blur_axis(result, r, 1)
    return result


def lift_albedo(albedo, mask, options):
    """把遮罩区域提亮并染色，同时保留原来的明暗细节。

    关键点：**整体明暗由低频（模糊后）亮度决定，高频细节按原始幅度叠加回去。**

    一开始的做法是直接 ``原亮度 / detail_max`` 当明暗系数，结果在噪声多的贴图上
    把高频噪声放大了好几倍（FumeUltra 那把烟熏大剑整张糊成雪花点）。
    拆成低频+高频之后就干净了：低频给出平滑的金属底色，
    高频（划痕、磨损）保持原来的幅度叠加，不再被放大。
    """
    fill = np.array(options.fill, dtype=np.float32)
    lum = luminance(albedo)
    base_src = smooth_luminance(lum, options.detail_blur)

    norm = np.clip(base_src / max(options.detail_max, 1e-6), 0.0, 1.0)
    base = options.floor + (1.0 - options.floor) * norm
    detail = (lum - base_src) * options.detail_gain

    lifted = np.clip(fill[None, None, :] * (base[:, :, None] + detail[:, :, None]), 0.0, 1.0)
    m = mask[:, :, None]
    return albedo * (1.0 - m) + lifted * m


def process_image(albedo_path, reflectivity_path, out_path, options, stats):
    albedo = load_rgb(albedo_path)
    reflectivity = load_rgb(reflectivity_path) if reflectivity_path else None

    mask = build_mask(albedo, reflectivity, options, stats)
    coverage = float(mask.mean())
    stats["coverage"] = coverage

    if options.feather > 0:
        mask = box_blur_mask(mask, options.feather)

    result = lift_albedo(albedo, mask, options)

    stats["before_mean"] = float(luminance(albedo).mean())
    stats["after_mean"] = float(luminance(result).mean())
    if coverage > 1e-6:
        sel = mask > 0.5
        if sel.any():
            stats["masked_before"] = float(luminance(albedo)[sel].mean())
            stats["masked_after"] = float(luminance(result)[sel].mean())

    if not options.dry_run:
        save_rgb(result, out_path, options.quality)
    return result


def make_preview(pairs, out_path, max_size=1400):
    """把 原图 | 处理后 横向拼成一张对比图，方便肉眼判断能不能接受。"""
    rows = []
    for before, after in pairs:
        a = Image.fromarray((np.clip(before, 0, 1) * 255).astype(np.uint8), "RGB")
        b = Image.fromarray((np.clip(after, 0, 1) * 255).astype(np.uint8), "RGB")
        rows.append((a, b))
    if not rows:
        return None
    cell = max(1, max_size // (2 * max(1, len(rows))))
    cell = min(cell, rows[0][0].width)
    sheet = Image.new("RGB", (cell * 2 + 12, cell * len(rows) + 6 * (len(rows) - 1)), (24, 25, 28))
    for i, (a, b) in enumerate(rows):
        a = a.resize((cell, cell), Image.LANCZOS)
        b = b.resize((cell, cell), Image.LANCZOS)
        y = i * (cell + 6)
        sheet.paste(a, (0, y))
        sheet.paste(b, (cell + 12, y))
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# 目录级流程
# --------------------------------------------------------------------------

def load_scene(folder):
    scene_path = os.path.join(folder, "scene.json")
    if not os.path.isfile(scene_path):
        return None
    with open(scene_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def process_folder(folder, options):
    """处理一个已解包目录，返回 (原贴图 -> 新贴图 的映射, 统计信息列表)。"""
    scene = load_scene(folder)
    if scene is None:
        raise SystemExit("目录里没有 scene.json：%s" % folder)

    mapping = {}
    reports = []
    previews = []

    # albedoTex 可能被多个材质共用，同一个文件只处理一次
    done = {}
    for material in scene.get("materials") or []:
        albedo_name = material.get("albedoTex")
        if not albedo_name:
            continue
        if albedo_name in done:
            mapping[albedo_name] = done[albedo_name][0]
            continue

        albedo_path = os.path.join(folder, albedo_name)
        if not os.path.isfile(albedo_path):
            reports.append((material.get("name"), albedo_name, None, "找不到贴图文件"))
            continue

        reflect_name = material.get("reflectivityTex")
        reflect_path = os.path.join(folder, reflect_name) if reflect_name else None
        if reflect_path and not os.path.isfile(reflect_path):
            reflect_path = None

        stem, ext = os.path.splitext(albedo_name)
        out_name = "%s%s%s" % (stem, options.suffix, ext or ".png")
        out_path = os.path.join(folder, out_name)

        stats = {}
        result = process_image(albedo_path, reflect_path, out_path, options, stats)

        if options.preview:
            previews.append((load_rgb(albedo_path), result))

        done[albedo_name] = (out_name, stats)
        mapping[albedo_name] = out_name
        reports.append((material.get("name"), albedo_name, out_name, stats))

    preview_path = None
    if options.preview and previews:
        preview_path = options.preview_path or os.path.join(folder, "_metal_fix_preview.png")
        make_preview(previews, preview_path)

    return mapping, reports, preview_path


def describe(reports, options):
    print("  %-22s %-18s %-8s %-10s %s" % ("材质", "反照率", "遮罩占比", "提亮前", "提亮后"))
    print("  " + "-" * 76)
    for name, albedo, out_name, stats in reports:
        if not isinstance(stats, dict):
            print("  %-22s %-18s 跳过：%s" % (str(name)[:22], albedo, stats))
            continue
        cov = stats.get("coverage", 0.0)
        before = stats.get("masked_before")
        after = stats.get("masked_after")
        extra = ""
        if "otsu" in stats:
            extra = "  Otsu=%.3f 分离度=%.2f" % (stats["otsu"], stats["separation"])
        print("  %-22s %-18s %6.1f%%  %-10s %-10s%s"
              % (str(name)[:22], albedo[:18], cov * 100.0,
                 "%.4f" % before if before is not None else "-",
                 "%.4f" % after if after is not None else "-", extra))
        if stats.get("note"):
            print("      ! %s" % stats["note"])


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fix_metal_albedo.py",
        description="把 PBR 反照率里全黑的金属区域提亮，方便在非 PBR 引擎里显示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  fix_metal_albedo.py 解包目录 --preview\n"
               "  fix_metal_albedo.py 解包目录 --strategy reflectivity --dry-run\n"
               "  fix_metal_albedo.py 解包目录 --fill 1,0.85,0.4 --floor 0.5\n")
    parser.add_argument("folder", help="已解包的目录（含 scene.json 和贴图）")
    parser.add_argument("--strategy", choices=("threshold", "reflectivity"), default="threshold",
                        help="金属识别策略（默认 threshold：按反照率亮度；"
                             "reflectivity：用 _r 贴图当遮罩）")
    parser.add_argument("--threshold", type=float, default=0.06,
                        help="threshold 策略的亮度阈值，低于它就算太黑（默认 0.06）")
    parser.add_argument("--soft", type=float, default=0.15,
                        help="软阈值过渡宽度（默认 0.15）。这一项对观感影响最大："
                             "太窄会在噪声贴图上出现椒盐斑点，"
                             "默认值配合 --threshold 0.06 相当于 0~0.21 平滑过渡")
    parser.add_argument("--reflectivity-threshold", default="auto",
                        help="reflectivity 策略的阈值，数字或 auto（默认 auto = 大津法）")
    parser.add_argument("--fill", default="1,1,1",
                        help="填充色，形如 R,G,B，取值 0~1（默认 1,1,1 纯白）")
    parser.add_argument("--floor", type=float, default=0.8,
                        help="提亮后的最低亮度比例（默认 0.8）")
    parser.add_argument("--detail-max", type=float, default=0.2,
                        help="低频亮度到多少就算'最亮'（默认 0.2）")
    parser.add_argument("--saturation-protect", type=float, default=0.7,
                        help="彩色像素的提亮折扣 0~1（默认 0.7）。"
                             "盾牌上金色/青色花纹的亮度也很低，只按亮度判断会被洗掉；"
                             "按饱和度打折就能保住图案。设 0 关闭")
    parser.add_argument("--detail-blur", type=float, default=8.0,
                        help="低频/高频的分界半径，像素（默认 8；设 0 则不分离，"
                             "噪声多的贴图会被放大成雪花点）")
    parser.add_argument("--detail-gain", type=float, default=1.0,
                        help="叠加回去的高频细节幅度（默认 1.0 = 保持原幅度；"
                             "设 0 得到完全平滑的金属）")
    parser.add_argument("--feather", type=float, default=0.0,
                        help="遮罩羽化半径（像素，默认 0 不羽化）")
    parser.add_argument("--suffix", default="_fixed", help="输出文件名后缀（默认 _fixed）")
    parser.add_argument("--quality", type=int, default=95, help="JPEG 输出质量（默认 95）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    parser.add_argument("--preview", action="store_true", help="额外生成左右对比图")
    parser.add_argument("--preview-path", help="对比图的输出路径")
    parser.add_argument("--print-mapping", action="store_true",
                        help="以 JSON 打印 原贴图->新贴图 的映射（供 wview_tool.py 调用）")
    return parser


def selftest():
    """自检：用合成图验证变换的数学行为，不依赖外部样本。"""
    checks = []

    def check(label, ok, detail=""):
        checks.append((label, bool(ok), detail))
        print("  [%s] %s%s" % ("通过" if ok else "失败", label,
                               ("  -- " + detail) if detail else ""))

    # --- 软阈值 ---
    check("soft_below: 纯黑取到满值 1.0",
          abs(float(soft_below(np.array([0.0]), 0.06 - 0.15, 0.06 + 0.15)[0]) - 1.0) < 1e-6)
    check("soft_below: 高于上界为 0",
          float(soft_below(np.array([0.5]), -0.09, 0.21)[0]) == 0.0)
    check("soft_below: 单调不增",
          bool(np.all(np.diff(soft_below(np.linspace(0, 1, 200), 0.0, 0.21)) <= 1e-7)))
    check("soft_above: 高于上界为 1",
          float(soft_above(np.array([0.9]), 0.3, 0.5)[0]) == 1.0)
    check("soft=0 时退化为硬阈值",
          float(soft_below(np.array([0.05]), 0.06, 0.06)[0]) == 1.0 and
          float(soft_below(np.array([0.07]), 0.06, 0.06)[0]) == 0.0)

    # --- 大津法 ---
    bimodal = np.concatenate([np.full(1000, 0.2), np.full(1000, 0.6)]).astype(np.float32)
    t, sep = otsu(bimodal)
    check("otsu 能把双峰分开", 0.2 < t < 0.6 and sep > 0.3, "阈值=%.3f 分离度=%.3f" % (t, sep))
    single = np.full(1000, 0.5, dtype=np.float32)
    t2, sep2 = otsu(single)
    check("otsu 对单峰分布的分离度很低", sep2 < 0.05, "分离度=%.3f" % sep2)

    # --- 提亮变换 ---
    options = argparse.Namespace(
        strategy="threshold", threshold=0.06, soft=0.15,
        reflectivity_threshold="auto", fill=[1.0, 1.0, 1.0],
        floor=0.8, detail_max=0.2, detail_blur=0.0, detail_gain=0.0,
        saturation_protect=0.0, feather=0.0, suffix="_fixed", quality=95,
        dry_run=True, preview=False, preview_path=None, print_mapping=False,
    )
    # 左半纯黑、右半彩色（金色），带一点噪声
    rng = np.random.default_rng(7)
    albedo = np.zeros((64, 64, 3), dtype=np.float32)
    albedo[:, :32] = 0.0
    albedo[:, 32:] = np.array([0.75, 0.62, 0.2], dtype=np.float32)
    albedo[:, :32] += (rng.random((64, 32, 1)) * 0.02).astype(np.float32)

    stats = {}
    mask = build_mask(albedo, None, options, stats)
    result = lift_albedo(albedo, mask, options)

    black_before = float(luminance(albedo)[:, :32].mean())
    black_after = float(luminance(result)[:, :32].mean())
    gold_before = float(luminance(albedo)[:, 32:].mean())
    gold_after = float(luminance(result)[:, 32:].mean())

    check("黑色区域被显著提亮", black_after > black_before + 0.4,
          "%.4f -> %.4f" % (black_before, black_after))
    check("金色区域基本不动", abs(gold_after - gold_before) < 0.02,
          "%.4f -> %.4f" % (gold_before, gold_after))
    check("输出仍在 0~1 之间",
          float(result.min()) >= 0.0 and float(result.max()) <= 1.0,
          "范围 %.3f~%.3f" % (result.min(), result.max()))

    # 饱和度保护应让金色区域遮罩更小
    options.saturation_protect = 0.9
    guard_mask = build_mask(albedo, None, options, {})
    check("饱和度保护降低了彩色区域的遮罩值",
          float(guard_mask[:, 32:].mean()) < float(guard_mask[:, :32].mean()) * 0.5,
          "彩色=%.3f 黑色=%.3f" % (guard_mask[:, 32:].mean(), guard_mask[:, :32].mean()))

    # 高频/低频分离：同一片带噪声的纯黑区域，用模糊后的低频版本算亮度，
    # 得到的表面应当平滑得多。这里特意用**整张纯黑**的图，避免黑白交界的
    # 模糊渗透干扰测量。
    options.saturation_protect = 0.0
    options.detail_gain = 0.0
    noisy = (rng.random((64, 64, 1)) * 0.05).astype(np.float32)
    noisy = np.repeat(noisy, 3, axis=2)

    def spread_of(blur):
        options.detail_blur = blur
        m = build_mask(noisy, None, options, {})
        out = luminance(lift_albedo(noisy, m, options))
        return float(np.abs(out - out.mean()).max())

    spread_raw = spread_of(0.0)
    spread_blurred = spread_of(8.0)
    check("低频/高频分离显著减小了噪声放大",
          spread_blurred < spread_raw * 0.5,
          "不分离=%.4f 分离后=%.4f" % (spread_raw, spread_blurred))
    options.detail_blur = 8.0

    # --- 输出编码往返 ---
    tmpdir = tempfile.mkdtemp(prefix="metal_selftest_")
    try:
        for ext in (".jpg", ".png"):
            path = os.path.join(tmpdir, "t" + ext)
            save_rgb(result, path, 95)
            back = load_rgb(path)
            check("写出并读回 %s 尺寸一致" % ext, back.shape == result.shape)
            check("%s 往返误差很小" % ext,
                  float(np.abs(back - result).mean()) < 0.01,
                  "平均误差=%.4f" % float(np.abs(back - result).mean()))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    passed = sum(1 for _l, ok, _d in checks if ok)
    print("\n通过 %d / %d" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if "--selftest" in sys.argv:
        return selftest()

    parser = build_parser()
    options = parser.parse_args()

    try:
        options.fill = [float(x) for x in str(options.fill).split(",")]
    except ValueError:
        parser.error("--fill 需要形如 1,1,1 的三个数字")
    if len(options.fill) != 3:
        parser.error("--fill 需要正好三个数字")
    if not (0.0 <= options.floor <= 1.0):
        parser.error("--floor 需要在 0~1 之间")

    if options.reflectivity_threshold != "auto":
        try:
            options.reflectivity_threshold = float(options.reflectivity_threshold)
        except ValueError:
            parser.error("--reflectivity-threshold 需要是数字或 auto")

    if not os.path.isdir(options.folder):
        print("错误：不是目录：%s" % options.folder)
        return 2

    mapping, reports, preview_path = process_folder(options.folder, options)

    if not options.print_mapping:
        print("目录：%s" % options.folder)
        print("策略：%s%s" % (options.strategy,
                             "（无 _r 贴图已退回 threshold）"
                             if getattr(options, "strategy_used", "") == "threshold"
                             and options.strategy == "reflectivity" else ""))
        describe(reports, options)
        if options.dry_run:
            print("\n--dry-run：只统计，没有写出任何文件。")
        else:
            print("\n已写出 %d 个提亮后的贴图（原图未改动）" % len(mapping))
            for old, new in mapping.items():
                print("  %s  ->  %s" % (old, new))
        if preview_path:
            print("对比图：%s" % preview_path)
    else:
        print(json.dumps(mapping, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
