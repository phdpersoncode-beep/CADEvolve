import cadquery as cq
import math

R = 20
dy = 4
cell_size = 3
x_off = -2
r_mid = 8
phi = math.pi / 4

points_a = []
for i in range(2):
    y = i * dy
    for j in range(2):
        x = j * cell_size + x_off
        z = math.sqrt(R * R - x * x - y * y)
        points_a.append((x, y))

points_b = []
for i in range(2):
    y = -i * dy
    for j in range(2):
        x = r_mid * math.cos(phi + j * math.pi)
        z = math.sqrt(R * R - x * x - y * y)
        points_b.append((x, y))

all_points = points_a + points_b
result = cq.Workplane("XY").pushPoints(all_points).circle(0.5).extrude(2)
