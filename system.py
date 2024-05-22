import tkinter as tk
from tkinter import Frame

def create_home_layout(frame):
    canvas = tk.Canvas(frame, width=600, height=400)
    canvas.pack()

    # 创建图形并绑定点击事件
    canvas.create_rectangle(50, 50, 150, 100, fill="yellow", tags="camera")
    canvas.create_text(100, 75, text="摄像头")
    canvas.create_rectangle(200, 50, 300, 100, fill="green", tags="detection")
    canvas.create_text(250, 75, text="检测")
    canvas.create_rectangle(350, 50, 450, 100, fill="red", tags="cutting")
    canvas.create_text(400, 75, text="切割")

    canvas.tag_bind("camera", "<Button-1>", lambda x: show_camera(frame))
    canvas.tag_bind("detection", "<Button-1>", lambda x: show_detection(frame))
    canvas.tag_bind("cutting", "<Button-1>", lambda x: show_cutting(frame))

def show_home(frame):
    clear_frame(frame)
    create_home_layout(frame)

def show_camera(frame):
    clear_frame(frame)
    tk.Label(frame, text="摄像头部分的编辑界面").pack()
    tk.Button(frame, text="返回主页", command=lambda: show_home(frame)).pack()

def show_detection(frame):
    clear_frame(frame)
    tk.Label(frame, text="检测部分的编辑界面").pack()
    tk.Button(frame, text="返回主页", command=lambda: show_home(frame)).pack()

def show_cutting(frame):
    clear_frame(frame)
    tk.Label(frame, text="切割部分的编辑界面").pack()
    tk.Button(frame, text="返回主页", command=lambda: show_home(frame)).pack()

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

# 创建主窗口
window = tk.Tk()
window.title("白萝卜缨检测切割系统")

# 创建主内容框架
main_frame = Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True)

# 初始化主页布局
show_home(main_frame)

window.mainloop()
