from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
from kivy.config import Config


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
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]
        print(touch.x)
        print(touch.y)
        # MyPaintWidget.coordinates.append([touch.x, touch.y])


class MyPaintApp(App):

    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    Config.set('graphics', 'resizable', 'false')  # 0 being off 1 being on as in true/false
    Config.set('graphics', 'width', '560')
    Config.set('graphics', 'height', '560')
    Config.write()
    MyPaintApp().run()
    print("coordinates below")
    print(MyPaintWidget.coordinates)