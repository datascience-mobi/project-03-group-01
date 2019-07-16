# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.widget import Widget
# from kivy.graphics import Color, Line
# from kivy.core.window import Window
# from kivy.config import Config
# import kivy.uix.image as image
# from kivy.uix.button import Button
# from random import randint
import image_operations
from PIL import Image, ImageFilter


def image_prepare(path) -> list:
    """
    transforms input image as mnist compatible intensity list
    :param path: image path
    :return: transformed image as list
    """
    # open input image from path
    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])

    # creates white canvas of 28x28 pixels
    new_image = Image.new('L', (28, 28), 255, color=0)

    # check which dimension is bigger, redundant as long as canvas width = height -> Not tested
    if width > height:
        # Width is bigger. Width becomes 20 pixels.
        new_height = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if new_height == 0:  # rare case but minimum is 1 pixel
            new_height = 1

        # resize and sharpen
        img = im.resize((20, new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        w_top = int(round(((28 - new_height) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, w_top))  # paste resized image on white canvas
    else:
        # TODO mnist image should have black margin instead of none?
        # Height is bigger. height becomes 20 pixels.
        new_width = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if new_width == 0:  # rare case but minimum is 1 pixel
            new_width = 1
            # resize and sharpen
        img = im.resize((new_width, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        w_left = int(round(((20 - new_width) / 2), 0))  # calculate vertical position TODO probably shifts pixels?
        new_image.paste(img, (w_left, 0))  # paste resized image on white canvas

    # save mnist styled image to list
    mnist_image = list(new_image.getdata())  # get pixel values

    print(mnist_image)
    return mnist_image


# # Canvas widget for drawing window
# class MyPaintWidget(Widget):
#
#     def __init__(self, **kwargs):
#         """
#         Setting up the key recognition and the drawing widget within the window
#         :param kwargs: TODO probably has to be kept because it's a prebuild kivy function
#         """
#         super(MyPaintWidget, self).__init__(**kwargs)
#         self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
#         self._keyboard.bind(on_key_down=self._on_keyboard_down)
#
#     def _keyboard_closed(self):
#         self._keyboard.unbind(on_key_down=self._on_keyboard_down)
#         self._keyboard = None
#
#     def _on_keyboard_down(self, keyboard, key_code, text, modifiers) -> bool:
#         """
#         Specifying key events
#         :param keyboard: not used TODO probably has to be kept because it's a prebuild kivy function
#         :param key_code: which key is pressed
#         :param text: not used TODO probably has to be kept because it's a prebuild kivy function
#         :param modifiers: not used TODO probably has to be kept because it's a prebuild kivy function
#         :return: True if no error occured
#         """
#         if key_code[1] == 'enter':  # Apply drawn canvas on enter key pressed
#             print(str("Leaving canvas, saving entered digit .."))
#             MyPaintApp.get_running_app().stop()
#             MyPaintWidget.export_to_png(self, "test.png")
#         elif key_code[1] == 'escape':  # Reset Canvas to restart drawing on escape key pressed
#             print(str("Resetting canvas .."))
#             self.canvas.clear()
#         return True
#
#     def on_touch_down(self, touch) -> None:
#         """
#         Begins line when left mouse button is first pressed
#         :param touch: contains cursor coordinates amongst others
#         :return: None
#         """
#         with self.canvas:
#             Color(1, 1, 1)
#             touch.ud['line'] = Line(points=(touch.x, touch.y), width=35)
#
#     def on_touch_move(self, touch) -> None:
#         """
#         Attaches coordinates to line when pressed mouse is moved
#         :param touch: contains cursor coordinates amongst others
#         :return: None
#         """
#         touch.ud['line'].points += [touch.x, touch.y]


# TODO probably creates the canvas window
# class MyPaintApp(App):
#
#     def __init__(self, title):
#         App.__init__(self)
#         self.title="Please draw a "+str(title)
#
#     def build(self):
#         super_box = BoxLayout(orientation='vertical')
#         super_box.add_widget(MyPaintWidget())
#         # button = Button(text="hi", size_hint=(0.1, .1))
#         # super_box.add_widget(button)
#         try:
#             # load the image
#             picture = image(source="mnist.png")
#             # add to the main field
#             super_box.add_widget(picture)
#         except Exception as e:
#             print('Pictures: Unable to load <%s>' % "fname.png")
#
#         return super_box


# def drawn_image(digit) -> list:
#     """
#     Initializes draw_canvas
#     :return: mnist compatible drawn image
#     """
#     # Sets window parameters
#     Config.set('graphics', 'resizable', 'false')  # 0 being off 1 being on as in true/false
#     Config.set('graphics', 'width', '560')
#     Config.set('graphics', 'height', '560')
#     Config.write()
#
#     # Runs drawing window
#     MyPaintApp(digit).run()
#
#     # Transform saved image to mnist style image
#     x = image_prepare('test.png')  # path must fit path in MyPaintWidget.on_keyboard_down
#
#     # show and save mnist style input image
#     image_operations.draw(x)
#     image_operations.save("mnist.png", x)
#
#     # Control: does mnist style image actually fit mnist size?
#     print(len(x))  # mnist IMAGES are 28x28=784 pixels
#     return x


if __name__ == '__main__':
    # For debugging purposes -> directly call drawing functions without further recognition
    # random_digit = randint(0, 9)
    # print(random_digit)
    # y = drawn_image(random_digit)
    image_operations.draw(image_prepare("mnist.png"))
