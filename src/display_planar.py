from cs1lib import *
from planarsim import *

radius = 1

class TrajectoryView:
    def __init__(self, sampled_trajectory, center_x, center_y, scale):
        self.sampled_trajectory = sampled_trajectory
        self.center_x = center_x
        self.center_y = center_y
        self.scale = scale

    def draw(self):
        ox = None
        oy = None
        for i in range(len(self.sampled_trajectory)):
            frame = self.sampled_trajectory[i]
            x, y, theta = config_from_transform(frame)

            px = self.center_x + x * self.scale
            py = self.center_y - y * self.scale

            if ox == None:
                ox = px
                oy = py

            draw_line(ox, oy, px, py)

            ox = px
            oy = py


def display():
    clear()
    tview.draw()


if __name__ == '__main__':

    samples = sample_trajectory([controls_rs[0], controls_rs[3], controls_rs[5], controls_rs[3]], \
                           [1.0, 2.0, 2.0, 4.0], 9.0, 30)
    tview = TrajectoryView(samples, 400, 400, 40)

    start_graphics(display,width=800,height=800)
