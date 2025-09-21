# visualize_mat_folder.py
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import h5py

# ================== 可配置 ==================
ROOT_DIR = r"/home/lin/Bearing-Fault-Detection/data"           # 要遍历的 .mat 文件夹
OUT_DIR  = r"/home/lin/Bearing-Fault-Detection/data_preprocess/img/raw_data"       # 导出图片的文件夹
KEY_FILTER = None              # 例如 r"^(signal|data)$" 只画匹配的变量；None 为不过滤
MAX_VARS_PER_FILE = 6          # 每个文件最多可视化的变量数
MAX_2D_ROWS = 3                # 对 2D 数组，最多展示的行数
DPI = 150
# ===========================================

os.makedirs(OUT_DIR, exist_ok=True)
key_pattern = re.compile(KEY_FILTER) if KEY_FILTER else None


# ---------- 工具：h5py(v7.3+) 读取并转 numpy ----------
def _h5_to_dict(obj):
    """递归把 h5py 的组/数据集转为 Python dict / numpy"""
    if isinstance(obj, h5py.Dataset):
        # 标量字符串/字节做解码
        arr = obj[()]
        if isinstance(arr, bytes):
            return arr.decode("utf-8", errors="ignore")
        return np.array(arr)
    elif isinstance(obj, h5py.Group):
        return {k: _h5_to_dict(obj[k]) for k in obj.keys()}
    else:
        return obj

def load_mat_any(path):
    """
    自动兼容：
    - v7.2 及更早: scipy.io.loadmat
    - v7.3 及之后: h5py
    返回: dict 结构 (变量名 -> numpy/标量/嵌套 dict)
    """
    try:
        data = loadmat(path)
        # 去掉 MATLAB 自带的元数据键
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            return {k: _h5_to_dict(f[k]) for k in f.keys()}


# ---------- 工具：从 dict 提取数值 ndarray ----------
def flatten_vars(d, prefix=""):
    """
    把嵌套 dict 展平为 {full_key: value}
    仅收集 numpy 数组或可转 numpy 的列表
    """
    out = {}
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_vars(v, name))
        else:
            try:
                arr = np.array(v)
                # 至少要是数值类型
                if np.issubdtype(arr.dtype, np.number):
                    out[name] = arr
            except Exception:
                pass
    return out


# ---------- 可视化 ----------
def plot_array(arr, title, save_path):
    """
    简单规则：
    - 1D: 画折线
    - 2D: 取前 MAX_2D_ROWS 行分别画（横轴用列索引）
    - >2D: 展示其第0 slice 转为 2D 后的画法
    """
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        # 降到 2D (取第0切片)
        arr = arr[(0,) * (arr.ndim - 2)]
    plt.figure(figsize=(8, 3 if arr.ndim == 1 else 4), dpi=DPI)

    if arr.ndim == 1:
        plt.plot(arr)
        plt.xlabel("Index")
        plt.ylabel("Value")
    elif arr.ndim == 2:
        rows = min(arr.shape[0], MAX_2D_ROWS)
        cols = 1
        fig_rows = rows
        # 独立子图（每行一条曲线）
        plt.clf()
        fig_h = max(3.0, 2.0 * rows)
        plt.figure(figsize=(8, fig_h), dpi=DPI)
        for i in range(rows):
            ax = plt.subplot(fig_rows, cols, i + 1)
            ax.plot(arr[i, :])
            ax.set_ylabel(f"Row {i}")
            ax.set_xlabel("Index")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # 标量或空
        plt.text(0.5, 0.5, f"Shape: {arr.shape}", ha="center", va="center")
    plt.suptitle(title, y=1.02 if arr.ndim == 2 else 1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    mat_paths = []
    for root, _, files in os.walk(ROOT_DIR):
        for fn in files:
            if fn.lower().endswith(".mat"):
                mat_paths.append(os.path.join(root, fn))

    if not mat_paths:
        print(f"[WARN] 在 {ROOT_DIR} 下没有找到 .mat 文件")
        return

    for path in mat_paths:
        rel = os.path.relpath(path, ROOT_DIR)
        print(f"[INFO] 处理: {rel}")
        try:
            data = load_mat_any(path)
        except Exception as e:
            print(f"  [ERROR] 读取失败: {e}")
            continue

        vars_dict = flatten_vars(data)
        if not vars_dict:
            print("  [WARN] 未找到数值型变量")
            continue

        # 变量过滤 & 只取前若干个
        items = list(vars_dict.items())
        if key_pattern:
            items = [(k, v) for k, v in items if key_pattern.search(k)]
        items = items[:MAX_VARS_PER_FILE]

        # 为当前文件创建导出子目录
        out_subdir = os.path.join(OUT_DIR, os.path.dirname(rel))
        os.makedirs(out_subdir, exist_ok=True)

        base = os.path.splitext(os.path.basename(path))[0]
        for i, (k, arr) in enumerate(items, 1):
            safe_key = re.sub(r"[^\w\-.]+", "_", k)
            save_path = os.path.join(out_subdir, f"{base}__{i:02d}__{safe_key}.png")
            title = f"{rel} :: {k}  shape={np.squeeze(arr).shape}"
            try:
                plot_array(arr, title, save_path)
            except Exception as e:
                print(f"  [WARN] 绘制 {k} 失败: {e}")

    print(f"[DONE] 图片已导出到: {OUT_DIR}")

if __name__ == "__main__":
    main()
