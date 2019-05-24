import tkinter as tk
import cv2
import numpy as np

w = None
drawn_pixels = None

drawing = False # true if mouse is pressed
pt1_x, pt1_y = None, None
coordinates = list()


def uniq(input_list):
  output = list()
  for x in input_list:
    if x not in output:
      output.append(x)
  return output


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


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x, pt1_y=x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)
            if x >= 0 and x <= 280 and y >= 0 and y <= 280:
                coordinates.append([pt1_x, pt1_y])
            pt1_x, pt1_y = x, y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)


if __name__ == '__main__':
    img = np.zeros((280, 280, 3), np.uint8)
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw', line_drawing)

    while (1):
        cv2.imshow('test draw', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print(uniq(coordinates))
