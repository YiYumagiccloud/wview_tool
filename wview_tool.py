
# -*- coding: utf-8 -*-
"""Wview / Marmoset Toolbag 查看器包 (.mview) 解包 + OBJ 导出工具 —— 优化版 v2.0

这是 测试.py 的重写版本，保持「拖入 .mview 即可得到解包目录 + OBJ」的使用方式，
但修掉了原版的若干缺陷，并补齐了原版缺失的功能。

用法::

    python wview_tool.py 模型.mview                    # 解包 + 导出 OBJ
    python wview_tool.py 模型.mview 另一个.mview ...    # 批量
    python wview_tool.py 模型.mview --list             # 只看内容，不写文件
    python wview_tool.py 已解包目录                     # 只把 scene.json 转成 OBJ
    python wview_tool.py 模型.mview -o 输出目录 --apply-transform --flip-v --wires

设计原则：
  * 解包结果与旧版逐字节一致（见 verify_equivalence.py）。
  * 任何损坏输入都给出可读的中文报错，而不是静默写坏文件或抛 TypeError。
  * 不改动旧脚本，新逻辑独立成文件，可并行验证。
"""

import argparse
import json
import os
import re
import struct
import sys

__version__ = "2.0"

# --- .mview 容器常量 ---------------------------------------------------------

FLAG_COMPRESSED = 0x01          # 第 0 位 = 数据经过 LZ 压缩
ENTRY_HEADER = struct.Struct("<III")   # flags / 压缩后字节数 / 原始字节数

# 顶点记录：位置(12) + UV(8) + 12 字节未解明数据（实测不是法线，见 README）
VERTEX_STRIDE_BASE = 32
POSITION_BYTES = 12
UV_BYTES = 8
UV_OFFSET = POSITION_BYTES


class WviewError(Exception):
    """输入文件损坏或格式不受支持。"""


# --- 容器解析 ---------------------------------------------------------------

class Entry(object):
    """容器中的一个条目。"""

    __slots__ = ("name", "kind", "flags", "stored_size", "size", "offset")

    def __init__(self, name, kind, flags, stored_size, size, offset):
        self.name = name                  # 条目名（通常是文件名）
        self.kind = kind                  # MIME 风格的类型标记
        self.flags = flags                # 位标志
        self.stored_size = stored_size    # 文件中实际占用的字节数
        self.size = size                  # 解压后的字节数
        self.offset = offset              # 数据在文件中的起始偏移

    @property
    def compressed(self):
        return bool(self.flags & FLAG_COMPRESSED)


def read_cstring(data, pos):
    """读取以 0 结尾的字符串，返回 (字符串, 新偏移, 是否为合法 UTF-8 文本)。

    原版用 ``struct.unpack("<b", f.read(1))`` 逐字节读，然后用 ``chr()`` 拼字符串：
    遇到 >= 0x80 的字节会得到负数，``chr(-128)`` 会产生垃圾字符，非 ASCII 名字必然损坏；
    而且每字节一次 unpack + 一次函数调用，慢且读不到结尾时抛的是 struct.error。
    这里改成一次性 find + 解码，并给出明确报错。

    第三个返回值用来区分「真正的文件名」和「二进制垃圾碰巧没有提前遇到 0 字节」：
    样本里的条目名都是纯 ASCII，是 UTF-8 时能干净解码；解不出来说明这不是容器。
    """
    end = data.find(b"\x00", pos)
    if end < 0:
        raise WviewError("偏移 %d 处的字符串没有结尾的 0 字节，文件可能被截断。" % pos)
    raw = data[pos:end]
    try:
        return raw.decode("utf-8"), end + 1, True
    except UnicodeDecodeError:
        return raw.decode("latin-1"), end + 1, False


def parse_entries(data):
    """把整个容器切成条目列表。"""
    entries = []
    pos = 0
    total = len(data)
    while pos < total:
        entry_start = pos
        name, pos, name_is_text = read_cstring(data, pos)
        kind, pos, _ = read_cstring(data, pos)
        # 轻量健全性检查：容器里存的都是 "thumbnail.jpg" 这类文件名。
        # 没有这个检查，随便丢一个 JPEG/zip 进来也会被"解析"出一堆垃圾条目，
        # 最后报一句莫名其妙的"文件被截断"。
        if (not name or len(name) > 255 or not name_is_text
                or any(ch < " " for ch in name)):
            raise WviewError("偏移 %d 处的条目名 %r 不像合法文件名，这可能不是 .mview 文件。"
                             % (entry_start, name[:40]))
        if pos + ENTRY_HEADER.size > total:
            raise WviewError("条目 %r 的头部不完整：需要 12 字节，只剩 %d 字节。"
                             % (name, total - pos))
        flags, stored_size, size = ENTRY_HEADER.unpack_from(data, pos)
        pos += ENTRY_HEADER.size
        if pos + stored_size > total:
            raise WviewError("条目 %r 声明 %d 字节数据，但文件只剩 %d 字节（文件被截断）。"
                             % (name, stored_size, total - pos))
        entries.append(Entry(name, kind, flags, stored_size, size, pos))
        pos += stored_size

    if not entries:
        # 原版对空文件会在 readcstr 里抛 struct.error；这里明确说清楚。
        raise WviewError("文件里没有任何条目，不是有效的 .mview 容器（文件为空或格式不符）。")
    return entries


# --- LZ 解压 ----------------------------------------------------------------

def _emit(out, d, p, count, limit):
    """等价于逐字节的 ``for i in range(count): out[d+i] = out[p+i]``。

    原版是纯 Python 的逐字节复制。这里改用切片赋值，并在重叠（LZ77 的游程复制）
    时按「周期重复」展开，语义与逐字节正向复制完全一致
    （verify_equivalence.py 里用 4000 组随机参数对拍验证过）。

    实测整体解压只快约 10%：真正的瓶颈是每条序列一次的 Python 主循环，
    而不是循环体内的字节复制，所以这里没有再做更激进的内联优化
    —— 那样只能再快约 10%，却要把边界检查在多条分支里重复一遍。
    """
    if count <= 0:
        return d
    end = d + count
    if end > limit:
        raise WviewError("解压输出越界：需要 %d 字节，声明只有 %d 字节。" % (end, limit))
    if p >= d:
        raise WviewError("解压数据损坏：回溯偏移 %d 未指向已解压数据（当前已解压 %d 字节）。" % (p, d))
    avail = d - p
    if count <= avail:
        out[d:end] = out[p:p + count]
    else:
        # p..d 的字节按周期重复，正是逐字节正向复制的效果
        pattern = bytes(out[p:d])
        repeats = -(-count // avail)          # 向上取整
        out[d:end] = (pattern * repeats)[:count]
    return end


def lz_decompress(src, expected_size):
    """解压一个条目。

    算法与原版逐位对齐（verify_equivalence.py 对全部 5 个样本的每个条目
    做了逐字节比对），差别只在这三点：
      * 输出缓冲区按声明长度一次分配，越界立即报错（原版会 IndexError）；
      * 长度不符时抛 WviewError（原版返回 None，调用方 write(None) 得到难懂的 TypeError）；
      * 复制循环用切片，实测整体解压快约 10%（解压本来就不是这条流水线的瓶颈）。
    """
    if expected_size <= 0:
        raise WviewError("解压前的长度声明为 %d，不是有效值。" % expected_size)
    if not src:
        raise WviewError("压缩数据为空。")

    out = bytearray(expected_size)
    seq_start = [0] * 4096        # e[]
    seq_len = [0] * 4096          # f[]
    g = 256
    h = len(src)
    d = 0
    out[0] = src[0]
    d = 1
    k = 0
    l = 1
    r = 1

    while True:
        n = r + (r >> 1)
        if (n + 1) >= h:
            break
        m = src[n + 1]
        n = src[n]
        p = (m << 4 | n >> 4) if (r & 1) else ((m & 15) << 8 | n)

        if p < g:
            if p < 256:
                # 字面量
                m = d
                n = 1
                if d >= expected_size:
                    raise WviewError("解压输出越界：字面量写在第 %d 字节，声明只有 %d 字节。"
                                     % (d, expected_size))
                out[d] = p
                d += 1
            else:
                # 回溯引用
                m = d
                n = seq_len[p]
                p = seq_start[p]
                d = _emit(out, d, p, n, expected_size)
        elif p == g:
            # 新序列：复制当前序列 l 字节，再补一个首字节
            m = d
            n = l + 1
            p = k
            d = _emit(out, d, p, l, expected_size)
            if d >= expected_size:
                raise WviewError("解压输出越界：新序列写在第 %d 字节，声明只有 %d 字节。"
                                 % (d, expected_size))
            out[d] = out[k]
            d += 1
        else:
            break

        seq_start[g] = k
        seq_len[g] = l + 1
        g += 1
        k = m
        l = n
        if g >= 4096:
            g = 256
        r += 1

    if d != expected_size:
        raise WviewError("解压结果 %d 字节，与声明的 %d 字节不符；压缩数据可能已损坏。"
                         % (d, expected_size))
    return bytes(out)


def entry_payload(data, entry):
    """取出条目内容（必要时解压）。"""
    raw = data[entry.offset:entry.offset + entry.stored_size]
    if entry.compressed:
        return lz_decompress(raw, entry.size)
    if len(raw) != entry.size:
        raise WviewError("条目 %r 未压缩，但长度 %d 与声明的 %d 不符。"
                         % (entry.name, len(raw), entry.size))
    return raw


# --- 输出路径安全 -----------------------------------------------------------

_UNSAFE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_output_path(root, name):
    """把条目名映射到输出目录内的安全路径。

    原版直接 ``open("%s/%s" % (folder, name))``：条目名里出现 ``../`` 或绝对路径
    就会写到输出目录之外（zip-slip 类问题）。这里逐段清洗并复核最终路径。
    """
    cleaned = name.replace("\\", "/").lstrip("/")
    parts = []
    for part in cleaned.split("/"):
        if part in ("", ".", ".."):
            continue
        part = _UNSAFE_CHARS.sub("_", part).rstrip(" .")
        if part:
            parts.append(part)
    if not parts:
        parts = ["unnamed"]

    target = os.path.join(root, *parts)
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    try:
        inside = os.path.commonpath([root_real, target_real]) == root_real
    except ValueError:            # 不同盘符
        inside = False
    if not inside:
        raise WviewError("条目名 %r 会写到输出目录之外，已拒绝。" % name)
    return target


def write_bytes(path, blob):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as handle:
        handle.write(blob)


# --- 顶点流读取 -------------------------------------------------------------

def vertex_stride(mesh):
    """一条顶点记录占多少字节。

    基础 32 字节 = 位置(12) + UV(8) + 12 字节用途未明数据；
    带顶点色 +4，带第二套 UV +8。
    """
    stride = VERTEX_STRIDE_BASE
    if mesh.get("vertexColor", 0):
        stride += 4
    if mesh.get("secondaryTexCoord", 0):
        stride += 8
    return stride


def read_mesh_stream(folder, mesh, warnings):
    """读出索引、边索引、顶点位置与 UV。

    布局（对全部 5 个样本逐字节验证过，见 README）::

        [三角形索引 indexCount * indexTypeSize]
        [边/线框索引 wireCount * indexTypeSize]
        [顶点顶点 vertexCount * stride]

    原版假定子网格索引是紧挨着顺序排列的，忽略了 ``firstIndex`` /
    ``firstWireIndex``；样本数据恰好都是单个子网格且 firstIndex=0 才没出错。
    这里按声明偏移读取，多子网格文件也能正确处理。
    """
    path = os.path.join(folder, mesh["file"])
    with open(path, "rb") as handle:
        data = handle.read()

    vertex_count = mesh["vertexCount"]
    index_count = mesh["indexCount"]
    wire_count = mesh["wireCount"]
    index_size = mesh["indexTypeSize"]
    if index_size not in (2, 4):
        raise WviewError("%s：indexTypeSize=%r 不是受支持的 2 或 4。" % (mesh["file"], index_size))

    stride = vertex_stride(mesh)
    index_bytes = index_count * index_size
    wire_bytes = wire_count * index_size
    vertex_offset = index_bytes + wire_bytes
    expected = vertex_offset + vertex_count * stride
    if expected != len(data):
        warnings.append("%s：按 scene.json 推算应为 %d 字节，实际 %d 字节（相差 %+d），"
                        "可能是不受支持的网格变体。"
                        % (mesh["file"], expected, len(data), len(data) - expected))

    ifmt = "<H" if index_size == 2 else "<I"
    code = "H" if index_size == 2 else "I"
    indices = struct.unpack_from("<%d%s" % (index_count, code), data, 0) if index_count else ()
    wires = (struct.unpack_from("<%d%s" % (wire_count, code), data, index_bytes)
             if wire_count else ())

    positions = []
    uvs = []
    for v in range(vertex_count):
        base = vertex_offset + v * stride
        if base + POSITION_BYTES + UV_BYTES > len(data):
            raise WviewError("%s：读取第 %d 个顶点时超出文件末尾。" % (mesh["file"], v))
        positions.append(struct.unpack_from("<fff", data, base))
        uvs.append(struct.unpack_from("<ff", data, base + UV_OFFSET))

    # scene.json 里每条网格的 max(index) 恰好等于 vertexCount-1，
    # 这是校验「offset/stride 猜对了」最有力的证据。
    if indices:
        hi = max(indices)
        if hi >= vertex_count:
            warnings.append("%s：索引出现 %d，超过顶点数 %d，读出的几何可能不可信。"
                            % (mesh["file"], hi, vertex_count))
    if index_count % 3:
        warnings.append("%s：索引数 %d 不是 3 的倍数，末尾会有残缺三角形。"
                        % (mesh["file"], index_count))
    if wire_count % 2:
        warnings.append("%s：边索引数 %d 不是 2 的倍数。" % (mesh["file"], wire_count))

    return positions, uvs, indices, wires, index_size


def apply_transform(matrix, point):
    """Toolbag 的 transform 是列主序 4x4，平移在第 12/13/14 位。"""
    x, y, z = point
    return (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14])


# --- 材质/名称清洗 ----------------------------------------------------------

def sanitize_token(name, used):
    """把材质名/组名清洗成 OBJ 安全的记号。

    样本里 Sparda 的材质名是 ``NewMat01 (1)``、``NewMat01 (2)``，
    原版直接写进 ``newmtl`` / ``usemtl``：OBJ 规范不允许名字含空格，
    很多导入器会把空格后的内容截断，导致材质丢失或错配。
    """
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_") or "material"

    candidate = text
    counter = 1
    while candidate in used and used[candidate] != name:
        counter += 1
        candidate = "%s_%d" % (text, counter)
    used[candidate] = name
    return candidate


def build_material_names(materials, warnings):
    """生成 原材质名 -> OBJ 安全名 的映射。"""
    used = {}
    mapping = {}
    for material in materials:
        original = material.get("name") or "material"
        safe = sanitize_token(original, used)
        if safe != original:
            warnings.append("材质名 %r 含空格或特殊字符，OBJ 中改名为 %r。" % (original, safe))
        mapping[original] = safe
    return mapping


def write_mtl(folder, materials, mapping, albedo_override=None):
    """写 master.mtl。

    ``albedo_override`` 把原反照率文件名映射到"修复过金属黑区"的新文件名
    （由 fix_metal_albedo.py 生成），这样 MTL 会指向提亮后的贴图，原图保持不动。
    """
    albedo_override = albedo_override or {}
    lines = ["# 由 wview_tool.py v%s 生成\n" % __version__]
    for material in materials:
        original = material.get("name") or "material"
        if original != mapping[original]:
            lines.append("# 原材质名: %s\n" % original)
        lines.append("newmtl %s\n" % mapping[original])
        lines.append("Ka 1.000000 1.000000 1.000000\n")
        lines.append("Kd 1.000000 1.000000 1.000000\n")
        lines.append("Ks 1.000000 1.000000 1.000000\n")
        lines.append("Ns 100.000000\n")
        lines.append("d 1.000000\n")
        lines.append("illum 2\n")

        albedo = material.get("albedoTex")
        reflect = material.get("reflectivityTex")
        normal = material.get("normalTex")
        gloss = material.get("glossTex")
        extras = material.get("extrasTex")
        diffuse = albedo_override.get(albedo, albedo)

        if diffuse:
            lines.append("map_Ka %s\n" % diffuse)
            lines.append("map_Kd %s\n" % diffuse)
            if albedo in albedo_override:
                lines.append("# 上面这张是把金属黑区提亮后的版本，原图仍是 %s\n" % albedo)
        if reflect:
            lines.append("map_Ks %s\n" % reflect)
        if gloss:
            lines.append("map_Ns %s\n" % gloss)
        # 原版只导出 Ka/Kd，法线贴图被整条丢掉；这里补上，
        # 顺带补高光与光泽贴图。
        if normal:
            lines.append("map_Bump %s\n" % normal)
        if albedo and (material.get("blend", "none") != "none" or material.get("alphaTest", 0)):
            lines.append("map_d %s\n" % diffuse)
        if extras:
            # extrasTex 是打包的 AO / 自发光遮罩，不是标准 OBJ 通道，
            # 用注释记下来源，方便在 DCC 里手工接。
            lines.append("# extrasTex(打包 AO/自发光): %s\n" % extras)
        lines.append("\n")

    path = os.path.join(folder, "master.mtl")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))
    return path


# --- OBJ 导出 ---------------------------------------------------------------

def export_mesh(folder, mesh, mapping, options, warnings):
    positions, uvs, indices, wires, index_size = read_mesh_stream(folder, mesh, warnings)
    name = mesh.get("name") or mesh["file"]
    dat = mesh["file"]

    transform = mesh.get("transform")
    if options.apply_transform and transform:
        positions = [apply_transform(transform, p) for p in positions]

    used_groups = {}
    group = sanitize_token(name, used_groups)

    lines = ["mtllib master.mtl\n"]
    for point in positions:
        lines.append("v %r %r %r\n" % point)
    if options.flip_v:
        for uv in uvs:
            lines.append("vt %r %r\n" % (uv[0], 1.0 - uv[1]))
    else:
        for uv in uvs:
            lines.append("vt %r %r\n" % uv)

    submeshes = mesh.get("subMeshes") or []
    if not submeshes:
        submeshes = [{"material": None, "firstIndex": 0, "indexCount": len(indices),
                      "firstWireIndex": 0, "wireIndexCount": len(wires)}]

    skipped = 0
    triangle_total = 0
    for sub in submeshes:
        material = sub.get("material")
        safe_material = mapping.get(material, "material")
        first = sub.get("firstIndex", 0)
        count = sub.get("indexCount", 0)
        chunk = indices[first:first + count]
        lines.append("\ng %s\n" % group)
        lines.append("usemtl %s\n" % safe_material)
        for i in range(0, len(chunk) - 2, 3):
            a, b, c = chunk[i], chunk[i + 1], chunk[i + 2]
            if options.skip_degenerate and (a == b or b == c or a == c):
                skipped += 1
                continue
            triangle_total += 1
            a += 1
            b += 1
            c += 1
            lines.append("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, c, c, c))

    if options.wires and wires:
        lines.append("\ng %s_wire\n" % group)
        for sub in submeshes:
            first = sub.get("firstWireIndex", 0)
            count = sub.get("wireIndexCount", 0)
            chunk = wires[first:first + count]
            for i in range(0, len(chunk) - 1, 2):
                lines.append("l %d %d\n" % (chunk[i] + 1, chunk[i + 1] + 1))

    path = os.path.join(folder, "%s.obj" % dat)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))
    return path, triangle_total, skipped


def load_metal_module():
    """按文件路径加载同目录的 fix_metal_albedo.py。

    不写成顶层 import：那个模块依赖 numpy 和 Pillow，而本工具刻意保持零第三方依赖
    （打包出来的 exe 才几 MB）。只有真的用到 --fix-metal 时才尝试加载。
    """
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_metal_albedo.py")
    if not os.path.isfile(path):
        raise WviewError("找不到 fix_metal_albedo.py（应与 wview_tool.py 放在同一目录）。")
    spec = importlib.util.spec_from_file_location("fix_metal_albedo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_metal_fix(folder, options, warnings):
    """调用 fix_metal_albedo 提亮金属黑区，返回 {原贴图: 新贴图}。"""
    try:
        module = load_metal_module()
    except WviewError as error:
        warnings.append(str(error) + " 已跳过金属提亮。")
        return {}
    except ImportError as error:
        warnings.append("金属提亮需要 numpy 和 Pillow（%s），当前环境缺少，已跳过。" % error)
        return {}

    metal = argparse.Namespace(
        strategy=options.metal_strategy,
        threshold=options.metal_threshold,
        reflectivity_threshold=options.metal_reflectivity_threshold,
        fill=options.metal_fill,
        floor=options.metal_floor,
        detail_max=options.metal_detail_max,
        detail_blur=options.metal_detail_blur,
        detail_gain=options.metal_detail_gain,
        saturation_protect=options.metal_saturation_protect,
        soft=options.metal_soft,
        feather=options.metal_feather,
        suffix=options.metal_suffix,
        quality=options.metal_quality,
        dry_run=False,
        preview=options.metal_preview,
        preview_path=None,
        print_mapping=False,
        strategy_used=None,
    )
    if isinstance(metal.fill, str):
        metal.fill = [float(x) for x in metal.fill.split(",")]

    mapping, reports, preview_path = module.process_folder(folder, metal)
    if not options.quiet:
        print()
        print("  金属黑区提亮（策略 %s）：" % metal.strategy)
        module.describe(reports, metal)
        if preview_path:
            print("  对比图：%s" % preview_path)
    return mapping


def convert_folder(folder, options, warnings):
    """把已解包的目录转成 OBJ（对应 extract_model.py 的职责）。"""
    scene_path = os.path.join(folder, "scene.json")
    if not os.path.isfile(scene_path):
        return None
    # 原版用 open() 走系统默认编码，非 ASCII 的 scene.json 在中文/英文 Windows 上会
    # UnicodeDecodeError；这里显式按 UTF-8 读。
    with open(scene_path, "r", encoding="utf-8") as handle:
        scene = json.load(handle)

    materials = list(scene.get("materials") or [])

    # 子网格引用的材质未必都出现在 materials 里（手改过的 scene.json 很常见）。
    # 不补占位材质的话，OBJ 里会写出一个 master.mtl 中不存在的 usemtl，
    # 导入器要么报错要么静默丢材质。
    known = {m.get("name") for m in materials}
    referenced = set()
    for mesh in scene.get("meshes") or []:
        for sub in mesh.get("subMeshes") or []:
            if sub.get("material"):
                referenced.add(sub["material"])
    missing = sorted(referenced - known)
    if missing:
        warnings.append("scene.json 里有 %d 个被引用但未定义的材质：%s（已生成占位材质）"
                        % (len(missing), "、".join(repr(n) for n in missing[:5])))
        for name in missing:
            materials.append({"name": name})

    albedo_override = {}
    # 用 getattr：本函数也会被验证脚本直接调用，那些调用方不会构造完整的选项对象。
    if getattr(options, "fix_metal", False):
        albedo_override = run_metal_fix(folder, options, warnings)

    mapping = build_material_names(materials, warnings)
    mtl_path = write_mtl(folder, materials, mapping, albedo_override)

    obj_paths = []
    for mesh in scene.get("meshes") or []:
        if options.verbose:
            print("  转换 %s" % mesh.get("file"))
        path, triangles, skipped = export_mesh(folder, mesh, mapping, options, warnings)
        obj_paths.append(path)
        if options.verbose and skipped:
            print("     跳过退化三角形 %d 个" % skipped)
    return mtl_path, obj_paths


# --- 顶层流程 ---------------------------------------------------------------

def unpack_file(path, outdir, options, warnings):
    with open(path, "rb") as handle:
        data = handle.read()
    entries = parse_entries(data)

    # 先把所有目标路径算出来，再决定要不要动手。
    # 两个原因：
    #   1) 写到一半才发现某个文件已存在，会留下一个残缺的输出目录；
    #   2) 原来的写法用 os.path.exists 判断，会把「同一个容器里两条同名条目」
    #      也当成"已存在"而报错 —— 那两条本来应该后者覆盖前者（旧版就是这个行为）。
    targets = [(entry, safe_output_path(outdir, entry.name)) for entry in entries]

    # 覆盖策略：解包是确定性的（同一份输入永远得到同样的输出），所以
    #   * 目标文件内容和本次要写的内容**完全一样** -> 视为重复解包，直接放行（静默跳过写入）；
    #   * 内容**不一样** -> 这才叫覆盖，会真的丢数据，必须先整体拒绝。
    # 这样"重复拖同一个文件"和旧版一样顺手，同时又不会悄悄冲掉别的文件。
    # 先检查再写，避免写到一半才报错、留下残缺的输出目录。
    if not options.overwrite:
        conflicts = []
        seen = set()
        for entry, destination in targets:
            if destination in seen:
                continue
            seen.add(destination)
            if not os.path.exists(destination):
                continue
            try:
                with open(destination, "rb") as handle:
                    same = handle.read() == entry_payload(data, entry)
            except OSError:
                same = False
            if not same:
                conflicts.append(destination)
        if conflicts:
            raise WviewError(
                "以下 %d 个文件已存在且内容不同，本次未做任何改动"
                "（加 --overwrite 可覆盖）：\n    %s"
                % (len(conflicts), "\n    ".join(conflicts[:10])))

    written = []
    for entry, destination in targets:
        blob = entry_payload(data, entry)
        if not options.overwrite and os.path.exists(destination):
            with open(destination, "rb") as handle:
                if handle.read() == blob:
                    if not options.quiet:
                        print("  跳过 - %s（内容相同）" % entry.name)
                    continue
        write_bytes(destination, blob)
        written.append((entry, destination))
        if not options.quiet:
            suffix = "（解压 %d -> %d 字节）" % (entry.stored_size, entry.size) \
                if entry.compressed else ""
            print("  处理 - %s  类型 - %s%s" % (entry.name, entry.kind, suffix))
    return entries, written


def describe(path, options):
    with open(path, "rb") as handle:
        data = handle.read()
    entries = parse_entries(data)
    print("%s  共 %d 个条目，%d 字节" % (path, len(entries), len(data)))
    # 表头用 ASCII：中文是双宽字符，用 %-Ns 对齐反而会错位。
    print("  %-34s %-16s %-10s %-12s %s" % ("name", "type", "flags", "stored", "raw"))
    for entry in entries:
        print("  %-34s %-16s 0x%08x %-12d %d%s"
              % (entry.name, entry.kind, entry.flags, entry.stored_size, entry.size,
                 "  压缩" if entry.compressed else ""))
    return entries


def process(path, options, warnings):
    """处理一个输入（.mview 文件或已解包目录）。"""
    if os.path.isdir(path):
        if not options.quiet:
            print("转换已解包目录 %s ..." % path)
        result = convert_folder(path, options, warnings)
        if result is None:
            raise WviewError("%s 里没有 scene.json，无法转换成 OBJ。" % path)
        return

    if not os.path.isfile(path):
        raise WviewError("找不到文件：%s" % path)

    if options.list_only:
        describe(path, options)
        return

    # 与原版一致的输出目录规则：与输入同级的「去掉扩展名」目录。
    # 原版用 filename.split(".")[0]，遇到路径里带点（如 C:\a.b\m.mview）会切错。
    default_out = os.path.splitext(path)[0]
    outdir = options.out or default_out

    if not options.quiet:
        print()
        print("开始处理 %s 模型文件..." % os.path.basename(default_out))
        print()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    entries, _written = unpack_file(path, outdir, options, warnings)

    if options.no_obj:
        return
    if not any(entry.name == "scene.json" for entry in entries):
        if not options.quiet:
            print("  未找到 scene.json，跳过 OBJ 转换。")
        return
    if not options.quiet:
        print()
    convert_folder(outdir, options, warnings)


# --- 配置文件 ---------------------------------------------------------------

CONFIG_FILENAME = "wview_tool.config.json"

# 内置默认值。配置文件里没写的项就用这里的值。
BUILTIN_DEFAULTS = {
    "default_action": "unpack_and_obj",
    "output_dir": "",
    "apply_transform": False,
    "flip_v": False,
    "wires": False,
    "skip_degenerate": False,
    "fix_metal": False,
    "overwrite": False,
    "pause_on_finish": "auto",
    "metal": {
        "strategy": "threshold",
        "threshold": 0.06,
        "soft": 0.15,
        "reflectivity_threshold": "auto",
        "fill": [1.0, 1.0, 1.0],
        "floor": 0.8,
        "detail_max": 0.2,
        "detail_blur": 8.0,
        "detail_gain": 1.0,
        "saturation_protect": 0.7,
        "feather": 0.0,
        "suffix": "_fixed",
        "quality": 95,
        "preview": False,
    },
}

CONFIG_TEMPLATE = """{
  "_说明": "wview_tool.py 的配置文件。删掉本文件即恢复内置默认值。命令行参数优先级高于本文件。",
  "_default_action_可选值": ["unpack", "unpack_and_obj", "list", "ask"],

  "default_action": "unpack",
  "_default_action_含义": "unpack=只解包；unpack_and_obj=解包并导出OBJ；list=只列出内容；ask=每次都弹菜单",

  "output_dir": "",
  "_output_dir_含义": "留空表示与输入同级、去掉扩展名的目录",

  "apply_transform": false,
  "flip_v": false,
  "wires": false,
  "skip_degenerate": false,
  "overwrite": false,

  "pause_on_finish": "auto",
  "_pause_on_finish_可选值": ["auto", "always", "never"],
  "_pause_on_finish_含义": "auto=双击/拖放启动时才等待回车（默认），always=总是等，never=从不等",

  "fix_metal": false,
  "_fix_metal_说明": "把 PBR 反照率里全黑的金属区域提亮，方便在非 PBR 引擎里显示。需要 numpy 和 Pillow",

  "metal": {
    "_说明": "以下参数对应 fix_metal_albedo.py 的同名选项，详见该脚本 --help",
    "strategy": "threshold",
    "threshold": 0.06,
    "soft": 0.15,
    "reflectivity_threshold": "auto",
    "fill": [1.0, 1.0, 1.0],
    "floor": 0.8,
    "detail_max": 0.2,
    "detail_blur": 8.0,
    "detail_gain": 1.0,
    "saturation_protect": 0.7,
    "feather": 0.0,
    "suffix": "_fixed",
    "quality": 95,
    "preview": false
  }
}
"""


def config_search_paths(explicit=None):
    if explicit:
        return [explicit]
    candidates = [os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)]
    try:
        candidates.append(os.path.join(os.getcwd(), CONFIG_FILENAME))
    except OSError:
        pass
    return candidates


def load_config(explicit, warnings):
    """读取配置文件。返回 (配置字典, 实际使用的路径 或 None)。"""
    for path in config_search_paths(explicit):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError) as error:
                # 配置坏了不该让工具整个不能用，退回内置默认值并说清楚。
                warnings.append("配置文件 %s 读取失败（%s），已改用内置默认值。" % (path, error))
                return {}, None
            if not isinstance(data, dict):
                warnings.append("配置文件 %s 的顶层不是对象，已忽略。" % path)
                return {}, None
            return data, path
    if explicit:
        warnings.append("找不到指定的配置文件：%s" % explicit)
    return {}, None


def config_get(config, key, default):
    value = config.get(key, None)
    return default if value is None else value


def write_config_template(path, warnings):
    if os.path.exists(path):
        warnings.append("配置文件已存在，未覆盖：%s" % path)
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(CONFIG_TEMPLATE)
    return True


# --- 交互式菜单 -------------------------------------------------------------

MENU_CHOICES = [
    ("1", "解包 + 导出 OBJ（推荐）", {"action": "unpack_and_obj"}),
    ("2", "只解包（不生成 OBJ）", {"action": "unpack"}),
    ("3", "只看内容（列出条目，不写文件）", {"action": "list"}),
    ("4", "解包 + 导出 OBJ + 应用 transform", {"action": "unpack_and_obj", "apply_transform": True}),
    ("5", "解包 + 导出 OBJ + 提亮金属黑区", {"action": "unpack_and_obj", "fix_metal": True}),
    ("6", "解包 + 导出 OBJ（全部：transform + 金属提亮 + 线框）",
     {"action": "unpack_and_obj", "apply_transform": True, "fix_metal": True, "wires": True}),
    ("0", "退出", None),
]


def _console_will_close():
    """判断进程退出后控制台窗口会不会消失（双击 / 拖放启动）。

    原理：看当前控制台上挂着几个进程。只有自己一个，说明是系统为这个 exe
    新开的窗口，退出就没了；如果是从 cmd/PowerShell 里启动的，父进程也挂在
    同一个控制台上，数量 >= 2，就不需要多此一举地等回车。
    """
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        buffer = (ctypes.c_uint * 16)()
        count = kernel32.GetConsoleProcessList(buffer, 16)
        return 0 < count <= 1
    except Exception:
        return False


def ask_action(options, inputs, config, warnings):
    """打印菜单并让用户选一个操作，直接改在 options 上。"""
    default_action = config_get(config, "default_action", BUILTIN_DEFAULTS["default_action"])
    print()
    print("=" * 62)
    print(" Wview 解包工具 v%s" % __version__)
    print("=" * 62)
    if inputs:
        shown = "、".join(os.path.basename(p) for p in inputs[:4])
        if len(inputs) > 4:
            shown += " 等 %d 个" % len(inputs)
        print(" 待处理：%s" % shown)
    else:
        print(" 用法：把 .mview 文件拖到本程序（或 exe）上即可")
    print("-" * 62)
    for key, label, _payload in MENU_CHOICES:
        print("  %s) %s" % (key, label))
    print("-" * 62)
    print(" 直接回车 = 用配置里的默认操作（当前：%s）" % default_action)

    try:
        answer = input(" 请输入编号：").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not answer:
        return default_action

    for key, _label, payload in MENU_CHOICES:
        if answer == key:
            if payload is None:
                return None
            for name, value in payload.items():
                if name == "action":
                    options.action = value
                else:
                    setattr(options, name, value)
            return options.action

    warnings.append("无法识别的编号 %r，已改用默认操作。" % answer)
    return default_action


def apply_action(options, action):
    """把 default_action 翻译成具体开关。"""
    if action == "list":
        options.list_only = True
        options.no_obj = True
    elif action == "unpack":
        options.no_obj = True
    # unpack_and_obj / ask 都是默认的"解包 + 导出 OBJ"


def _add_toggle(parser, name, help_text):
    """加一对 --xxx / --no-xxx，未指定时 default 为 None（好和配置文件区分开）。"""
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, dest=dest, action="store_true", default=None,
                       help=help_text)
    group.add_argument("--no-" + name, dest=dest, action="store_false", default=None,
                       help="关闭上面的选项")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="wview_tool.py",
        description="Wview / Marmoset Toolbag .mview 解包与 OBJ 导出工具 v%s" % __version__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  wview_tool.py 模型.mview\n"
               "  wview_tool.py 模型.mview --list\n"
               "  wview_tool.py 模型.mview -o 输出 --apply-transform --flip-v --wires\n"
               "  wview_tool.py 模型.mview --fix-metal --metal-preview\n"
               "  wview_tool.py 已解包目录\n"
               "  wview_tool.py --ask                双击/拖放时弹出操作菜单\n"
               "  wview_tool.py --write-config       在当前目录生成配置文件模板\n")
    parser.add_argument("inputs", nargs="*", help=".mview 文件，或已解包的目录")
    parser.add_argument("-o", "--out", help="输出目录（多个输入时作为父目录）")
    parser.add_argument("-l", "--list", dest="list_only", action="store_true",
                        help="只列出容器内容，不写任何文件")
    parser.add_argument("--no-obj", action="store_true", help="只解包，不生成 OBJ/MTL")
    _add_toggle(parser, "apply-transform",
                "把 scene.json 里每个网格的 transform 应用到顶点上")
    _add_toggle(parser, "flip-v",
                "翻转 UV 的 V 方向（导入 Blender 等 Y 轴朝上的工具时常用）")
    _add_toggle(parser, "wires", "额外导出线框边为 OBJ 的 l 段")
    _add_toggle(parser, "skip-degenerate", "跳过三个索引有重复的退化三角形")
    _add_toggle(parser, "overwrite", "允许覆盖内容不同的已存在输出文件")
    _add_toggle(parser, "fix-metal",
                "把反照率里全黑的金属区域提亮（需要 numpy 和 Pillow），"
                "生成的贴图带 _fixed 后缀，原图不动，MTL 会自动指向新贴图")

    metal = parser.add_argument_group("金属提亮参数（对应 fix_metal_albedo.py）")
    metal.add_argument("--metal-strategy", choices=("threshold", "reflectivity"))
    metal.add_argument("--metal-threshold", type=float)
    metal.add_argument("--metal-soft", type=float)
    metal.add_argument("--metal-reflectivity-threshold")
    metal.add_argument("--metal-fill")
    metal.add_argument("--metal-floor", type=float)
    metal.add_argument("--metal-detail-max", type=float)
    metal.add_argument("--metal-detail-blur", type=float)
    metal.add_argument("--metal-detail-gain", type=float)
    metal.add_argument("--metal-saturation-protect", type=float)
    metal.add_argument("--metal-feather", type=float)
    metal.add_argument("--metal-suffix")
    metal.add_argument("--metal-quality", type=int)
    metal.add_argument("--metal-preview", action="store_true", default=None,
                       help="额外生成一张左右对比图，方便判断效果")

    config = parser.add_argument_group("配置文件与交互")
    config.add_argument("--config", help="指定配置文件路径（默认查找程序同目录的 %s）"
                        % CONFIG_FILENAME)
    config.add_argument("--write-config", action="store_true",
                        help="在当前目录生成配置文件模板后退出")
    config.add_argument("--action", choices=("unpack", "unpack_and_obj", "list", "ask"),
                        help="本次要执行的操作（覆盖配置文件里的 default_action）")
    config.add_argument("--ask", action="store_true", default=None,
                        help="即使带了文件参数也先弹出操作菜单")

    parser.add_argument("--strict", action="store_true",
                        help="把尺寸/索引异常当成错误而不是警告")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出更多细节")
    parser.add_argument("-V", "--version", action="version",
                        version="wview_tool.py v%s" % __version__)
    parser.add_argument("--pause", dest="pause", action="store_true", default=None,
                        help="结束前等待回车（双击/拖放启动时默认开启）")
    parser.add_argument("--no-pause", dest="pause", action="store_false",
                        help="结束前不等待")
    return parser


def resolve_options(options, config, warnings):
    """按 命令行 > 配置文件 > 内置默认 的优先级，把配置填进 options。"""
    def pick(cli_value, key, builtin):
        if cli_value is not None:
            return cli_value
        return config_get(config, key, builtin)

    options.apply_transform = pick(options.apply_transform, "apply_transform",
                                   BUILTIN_DEFAULTS["apply_transform"])
    options.flip_v = pick(options.flip_v, "flip_v", BUILTIN_DEFAULTS["flip_v"])
    options.wires = pick(options.wires, "wires", BUILTIN_DEFAULTS["wires"])
    options.skip_degenerate = pick(options.skip_degenerate, "skip_degenerate",
                                   BUILTIN_DEFAULTS["skip_degenerate"])
    options.overwrite = pick(options.overwrite, "overwrite", BUILTIN_DEFAULTS["overwrite"])
    options.fix_metal = pick(options.fix_metal, "fix_metal", BUILTIN_DEFAULTS["fix_metal"])
    if options.out is None:
        options.out = config_get(config, "output_dir", BUILTIN_DEFAULTS["output_dir"]) or None

    metal_config = config.get("metal") or {}
    if not isinstance(metal_config, dict):
        warnings.append("配置里的 metal 不是对象，已忽略。")
        metal_config = {}
    metal_defaults = BUILTIN_DEFAULTS["metal"]
    for name, default in metal_defaults.items():
        attr = "metal_" + name
        current = getattr(options, attr, None)
        if current is None:
            setattr(options, attr, metal_config.get(name, default))

    # 列表形式的填充色转成数值
    fill = options.metal_fill
    if isinstance(fill, str):
        try:
            fill = [float(x) for x in fill.split(",")]
        except ValueError:
            warnings.append("--metal-fill 需要形如 1,1,1 的三个数字，已改用默认白色。")
            fill = list(metal_defaults["fill"])
    if not isinstance(fill, (list, tuple)) or len(fill) != 3:
        warnings.append("metal.fill 需要正好三个数字，已改用默认白色。")
        fill = list(metal_defaults["fill"])
    options.metal_fill = [float(x) for x in fill]
    return options


def main(argv=None):
    # 原版固定 print 中文，在非中文代码页的控制台上会 UnicodeEncodeError 直接崩掉。
    # 让编码错误降级成替换字符，至少不会中断解包。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    parser = build_parser()
    options = parser.parse_args(argv)
    warnings = []

    if options.write_config:
        target = os.path.join(os.getcwd(), CONFIG_FILENAME)
        if write_config_template(target, warnings):
            print("已生成配置文件模板：%s" % target)
            print("把它放在 wview_tool.py / exe 旁边即可生效。")
        for text in warnings:
            sys.stderr.write("提示：%s\n" % text)
        return 0

    config, config_path = load_config(options.config, warnings)
    resolve_options(options, config, warnings)

    # 决定这次要做什么：命令行 --action > 配置文件 > 内置默认
    action = options.action
    if action is None:
        action = config_get(config, "default_action", BUILTIN_DEFAULTS["default_action"])
    if options.list_only:
        action = "list"
    if options.no_obj:
        action = "unpack"

    want_menu = (action == "ask") or bool(options.ask)
    if not options.inputs:
        # 双击启动、没带文件：弹菜单顺便当用法说明
        if not want_menu:
            parser.print_help()
            print("\n把 .mview 文件拖到本程序上即可开始；"
                  "或用 --write-config 生成配置文件改默认行为。")
            _finish_pause(options, config, had_inputs=False)
            return 2
        chosen = ask_action(options, [], config, warnings)
        for text in warnings:
            sys.stderr.write("提示：%s\n" % text)
        _finish_pause(options, config, had_inputs=False)
        return 0 if chosen is not None else 2

    if want_menu:
        chosen = ask_action(options, options.inputs, config, warnings)
        if chosen is None:
            print("已取消。")
            _finish_pause(options, config, had_inputs=True)
            return 0
        action = chosen

    apply_action(options, action)

    if config_path and options.verbose:
        print("配置文件：%s" % config_path)

    failures = 0
    multiple = len(options.inputs) > 1

    for path in options.inputs:
        sub_options = options
        if options.out and multiple and not os.path.isdir(path):
            # 多个输入共用一个输出目录时，各自放进以文件名命名的子目录
            sub_options = argparse.Namespace(**vars(options))
            sub_options.out = os.path.join(options.out, os.path.basename(os.path.splitext(path)[0]))
        try:
            process(path, sub_options, warnings)
        except WviewError as error:
            failures += 1
            sys.stderr.write("错误：%s\n" % error)
        except (OSError, struct.error, ValueError) as error:
            failures += 1
            sys.stderr.write("错误：处理 %s 时失败：%s: %s\n"
                             % (path, type(error).__name__, error))

    if warnings:
        sys.stderr.write("\n提示（%d 条）：\n" % len(warnings))
        for text in warnings:
            sys.stderr.write("  * %s\n" % text)
        if options.strict:
            failures += 1

    if not options.quiet and not failures:
        print()
        print("全部完成 !!!")

    _finish_pause(options, config, had_inputs=True)
    return 1 if failures else 0


def _finish_pause(options, config, had_inputs):
    """按需在退出前等回车，免得双击/拖放时窗口一闪而过。"""
    mode = options.pause
    if mode is None:
        mode = config_get(config, "pause_on_finish", BUILTIN_DEFAULTS["pause_on_finish"])
    if mode == "never" or mode is False:
        return
    if mode == "always" or mode is True:
        should_pause = True
    else:                      # auto
        should_pause = (not had_inputs) or _console_will_close()
    if not should_pause:
        return
    try:
        input("\n按回车键退出...")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    sys.exit(main())
