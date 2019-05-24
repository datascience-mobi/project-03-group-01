import tkinter as tk

w = None
drawn_pixels = None


def m_move(event) -> None:
    """
    Draws circle and saves mouse position to list drawn_pixels
    :param event: contains cursor position etc.
    :return: None
    """
    global drawn_pixels, w
    print(event.x, event.y)
    python_green = "#476042"
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    w.create_oval(x1, y1, x2, y2, fill=python_green)
    drawn_pixels.append([event.x, event.y])


def draw_pixel() -> None:
    """
    Sets up canvas, binds functions to events, runs canvas
    :return: None
    """
    global w, drawn_pixels
    root = tk.Tk()
    root.title("Hello")
    canvas_width = 28*10
    canvas_height = 28*10

    drawn_pixels = list()

    w = tk.Canvas(root, width=canvas_width, height=canvas_height)
    w.pack(expand=tk.YES, fill=tk.BOTH)

    root.bind('<B1-Motion>', m_move)
    root.bind("<Return>", lambda e: root.destroy())
    root.mainloop()


if __name__ == '__main__':
    draw_pixel()
    print(drawn_pixels)
