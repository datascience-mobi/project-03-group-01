from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
from kivy.config import Config
from PIL import Image, ImageFilter
import image_operations


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((28.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 0))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    #tva = [(255 - x) * 1.0 for x in tv]
    print(tv)
    return tv


class MyPaintWidget(Widget):

    # coordinates = list() # probably useless since the drawn digit is saved as png

    def __init__(self, **kwargs):
        """
        Setting up the key recognition and the drawing widget within the window
        :param kwargs:
        """
        super(MyPaintWidget, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, key_code, text, modifiers):
        """
        Specifying key events
        :param keyboard:
        :param key_code:
        :param text:
        :param modifiers:
        :return:
        """
        if key_code[1] == 'enter':
            print(str("Leaving canvas, saving entered digit .."))
            MyPaintApp.get_running_app().stop()
            MyPaintWidget.export_to_png(self,"test.png")
        elif key_code[1] == 'escape':
            print(str("Resetting canvas .."))
            self.canvas.clear()
            MyPaintWidget.coordinates = list()
        return True

    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=35)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]
        print(touch.x)
        print(touch.y)
        # MyPaintWidget.coordinates.append([touch.x, touch.y])


class MyPaintApp(App):

    def build(self):
        return MyPaintWidget()


def drawn_image() -> list:
    Config.set('graphics', 'resizable', 'false')  # 0 being off 1 being on as in true/false
    Config.set('graphics', 'width', '560')
    Config.set('graphics', 'height', '560')
    Config.write()
    MyPaintApp().run()
    # print("coordinates below")
    # print(MyPaintWidget.coordinates)
    x = imageprepare('test.png')  # file path here
    image_operations.draw(x)
    image_operations.save("mnist.png",x)
    print(len(x))  # mnist IMAGES are 28x28=784 pixels
    return x

if __name__ == '__main__':

    Config.set('graphics', 'resizable', 'false')  # 0 being off 1 being on as in true/false
    Config.set('graphics', 'width', '560')
    Config.set('graphics', 'height', '560')
    Config.write()
    MyPaintApp().run()
    # print("coordinates below")
    # print(MyPaintWidget.coordinates)
    x = imageprepare('test.png')  # file path here
    image_operations.draw(x)
    image_operations.save("mnist.png",x)
    print(len(x))  # mnist IMAGES are 28x28=784 pixels

