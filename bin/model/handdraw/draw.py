#This code isn't made by author but AI,because author had no idea to write a drawing gui.


import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import numpy as np

# ===== 配置 =====
CANVAS_SIZE = 280
IMG_SIZE = 28
BRUSH = 12

SAVE_DIR = r"D:\AIproj\bin\model\handdraw"      # ← 默认保存文件夹，改成你自己的
IMG_PREFIX = "digit"            # 文件名前缀
IMG_EXT = ".png"


class DrawApp:
    def __init__(self, root, on_submit=None):
        self.root = root
        self.on_submit = on_submit
        self.root.title("手写数字 28×28")

        # 默认文件夹不存在就创建
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 显示画布
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE,
                                height=CANVAS_SIZE, bg="black")
        self.canvas.pack()

        # PIL 图像（真正记录像素），模式 L = 灰度
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 鼠标事件：按下清空 last，移动画线，抬笔清空 last（防止连笔）
        self.canvas.bind("<Button-1>", self.reset_last)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)
        self.last_x, self.last_y = None, None

        # 按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="清空", command=self.clear).pack(side="left")
        tk.Button(btn_frame, text="保存(自动编号)",
                  command=self.save_auto).pack(side="left")
        tk.Button(btn_frame, text="另存为",
                  command=self.save_as).pack(side="left")
        tk.Button(btn_frame, text="处理", command=self.submit).pack(side="left")

    # ---------- 绘画 ----------
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="white", width=BRUSH,
                                    capstyle="round")
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=255, width=BRUSH)
        self.last_x, self.last_y = x, y

    def reset_last(self, event=None):
        """按下 / 抬起鼠标时调用，断开笔迹"""
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    # ---------- 共用：把画布转成 28×28 数组（裁剪 + 等比缩放 + 居中）----------
    def _to_28(self):
        # 1. 找数字边界框
        arr = np.array(self.image)              # (280, 280) uint8
        ys, xs = np.where(arr > 20)             # 非背景像素
        if len(xs) == 0:
            # 空白画布，返回全 0
            img28 = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
            return img28, np.array(img28, dtype=np.float32)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # 2. 裁剪出数字
        crop = self.image.crop((x0, y0, x1 + 1, y1 + 1))

        # 3. 等比缩放到最长边 = 20（留出 4 像素边距）
        w, h = crop.size
        scale = 20.0 / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

        # 4. 居中贴到 28×28 画布
        img28 = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        paste_x = (IMG_SIZE - new_w) // 2
        paste_y = (IMG_SIZE - new_h) // 2
        img28.paste(crop, (paste_x, paste_y))

        arr28 = np.array(img28, dtype=np.float32)
        return img28, arr28

    # ---------- 保存方式1：固定文件夹 + 自动编号 ----------
    def save_auto(self):
        img28, arr = self._to_28()

        # 找当前最大编号
        existing = [f for f in os.listdir(SAVE_DIR)
                    if f.startswith(IMG_PREFIX) and f.endswith(IMG_EXT)]
        idx = len(existing) + 1

        img_path = os.path.join(SAVE_DIR, f"{IMG_PREFIX}_{idx:03d}{IMG_EXT}")
        npy_path = os.path.join(SAVE_DIR, f"{IMG_PREFIX}_{idx:03d}.npy")

        img28.save(img_path)
        np.save(npy_path, arr)
        print(f"已保存第 {idx} 张:\n  {img_path}\n  {npy_path}")

    # ---------- 保存方式2：另存为（手动选路径） ----------
    def save_as(self):
        img28, arr = self._to_28()

        path = filedialog.asksaveasfilename(
            defaultextension=IMG_EXT,
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")],
            initialdir=SAVE_DIR,                    # 默认打开你的文件夹
            initialfile=f"{IMG_PREFIX}_custom{IMG_EXT}",
            title="另存为"
        )
        if not path:
            print("已取消保存")
            return

        img28.save(path)
        npy_path = path.rsplit(".", 1)[0] + ".npy"
        np.save(npy_path, arr)
        print(f"已另存为:\n  {path}\n  {npy_path}")

    def submit(self):
        img28, arr = self._to_28()
        if self.on_submit is not None:
            self.on_submit(arr)
        self.root.destroy()


def draw_and_get():
    result = {"arr": None}
    root = tk.Tk()
    app = DrawApp(root, on_submit=lambda arr: result.update(arr=arr))
    root.mainloop()
    return result["arr"]


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()