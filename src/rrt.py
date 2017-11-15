# using shapely for collision detection
# using annoy for ann search

from shapely.geometry import Polygon, Point

if __name__ == '__main__':
    poly = Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))
    point = Point(2, .2)

    print(poly)
    print(poly.contains(point))
