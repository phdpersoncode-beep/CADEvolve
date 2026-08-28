import cadquery as cq
housing_len = 80.0
housing_wid = 60.0
housing_ht = 40.0
wall_thick = 3.0
chamfer_size = 1.0
hole_diam = 6.0
hole_x = 0.0
hole_y = -20.0
channel_len = 70.0
channel_width = 5.0
channel_height = 8.0
channel_ampl = 8.0
# outer box
outer = cq.Workplane("XY").box(housing_len, housing_wid, housing_ht, centered=(True, True, False))
# shell to create thin walls
housed = outer.shell(-wall_thick)
# chamfer outer vertical edges
housed = housed.edges("|Z").chamfer(chamfer_size)
# mounting hole on top face
housed = housed.faces(">Z").workplane().center(hole_x, hole_y).hole(hole_diam)
# create sinusoidal path points in XZ plane
import math
points = []
steps = 20
for i in range(steps+1):
    x = -channel_len/2 + i*channel_len/steps
    z = channel_ampl * math.sin(2*math.pi * i/steps)
    points.append((x, z))
# path wire on XZ plane
path = cq.Workplane("XZ").spline(points)
# sweep rectangular profile along path to create channel solid
channel = cq.Workplane("YZ").rect(channel_width, channel_height).sweep(path)
# cut channel from housing
result = housed.cut(channel)
