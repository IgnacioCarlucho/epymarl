import os
import sys
import numpy as np
from gym import error
import six


if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


_BLACK = (0, 0, 0)


class Viewer:
    def __init__(self, world_size):
        display = get_display(None)

        # Let's have the display auto-scale to a 500x500 window
        self.rows, self.cols = world_size

        self.grid_size = 500 / self.rows
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1

        disp_height = 500
        disp_width = 500 * (self.cols / self.rows)

        self.window = pyglet.window.Window(
            width=int(disp_width), height=disp_height, display=display
        )

        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "assets")]
        pyglet.resource.reindex()

        self.agent_1 = pyglet.resource.image('agent_1.png')
        self.agent_2 = pyglet.resource.image('agent_2.png')
        self.goal_1 = pyglet.resource.image('1.png')
        self.goal_2 = pyglet.resource.image('2.png')
        self.wall = pyglet.resource.image('wall.png')

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(0.65, 0.65, 0.65, 0.65)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_agent(env)
        self._draw_shapes(env)

        if return_rgb_array:
            self.window.set_visible()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_agent(self, env):
        batch = pyglet.graphics.Batch()

        players = []

        for agent in env.agents:
            row, col = agent.y, agent.x
            if '0' in agent.id:
                players.append(
                    pyglet.sprite.Sprite(
                        self.agent_1,
                        self.grid_size * col,
                        self.height - self.grid_size * (row + 1),
                        batch=batch,
                    )
                )

            else:
                players.append(
                    pyglet.sprite.Sprite(
                        self.agent_2,
                        self.grid_size * col,
                        self.height - self.grid_size * (row + 1),
                        batch=batch,
                    )
                )

        for player in players:
            player.update(scale=self.grid_size / player.width)
        batch.draw()

    def _draw_shapes(self, env):
        batch = pyglet.graphics.Batch()

        walls = []

        for wall in env.walls:
            row, col = wall.y, wall.x
            walls.append(
                pyglet.sprite.Sprite(
                    self.wall,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )

        for w in walls:
            w.update(scale=self.grid_size / w.width)

        goals = []

        for goal in env.goals:
            row, col = goal.y, goal.x

            if '0' in goal.id:
                goals.append(
                    pyglet.sprite.Sprite(
                        self.goal_1,
                        self.grid_size * col,
                        self.height - self.grid_size * (row + 1),
                        batch=batch,
                    )
                )
            else:
                goals.append(
                    pyglet.sprite.Sprite(
                        self.goal_2,
                        self.grid_size * col,
                        self.height - self.grid_size * (row + 1),
                        batch=batch,
                    )
                )

        for g in goals:
            g.update(scale=self.grid_size / g.width)

        batch.draw()
