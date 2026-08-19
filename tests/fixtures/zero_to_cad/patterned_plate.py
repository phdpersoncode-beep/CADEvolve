import cadquery as cq
import math

plate_length = 100
plate_width = 80
plate_thickness = 8
chamfer_size = 0.8
pocket_length = 50
pocket_width = 30
pocket_depth = 4
small_hole_dia = 4
medium_hole_dia = 6
large_hole_dia = 8
hole_spacing_x = 20
hole_spacing_y = 20
hole_rows = 3
hole_cols = 4
hole_margin = 10
slot_length = 40
slot_width = 6
slot_angle_increment = 30
num_slots = 12
slot_radius = 30

result = (
    cq.Workplane("XY")
    .rect(plate_length, plate_width)
    .extrude(plate_thickness)
    .edges("|Z")
    .chamfer(chamfer_size)
)
result = (
    result.faces(">Z")
    .workplane()
    .center(0, 0)
    .rect(pocket_length, pocket_width)
    .cutBlind(pocket_depth)
)

small_hole_points = []
start_x = -plate_length / 2 + hole_margin + hole_spacing_x / 2
start_y = -plate_width / 2 + hole_margin + hole_spacing_y / 2
for i in range(hole_cols):
    for j in range(hole_rows):
        x = start_x + i * hole_spacing_x
        y = start_y + j * hole_spacing_y
        small_hole_points.append((x, y))
result = (
    result.faces(">Z")
    .workplane()
    .pushPoints(small_hole_points)
    .hole(small_hole_dia)
)

medium_hole_points = []
offset = hole_spacing_x / 2
for i in range(hole_cols - 1):
    for j in range(hole_rows - 1):
        x = start_x + offset + i * hole_spacing_x
        y = start_y + offset + j * hole_spacing_y
        medium_hole_points.append((x, y))
result = (
    result.faces(">Z")
    .workplane()
    .pushPoints(medium_hole_points)
    .hole(medium_hole_dia)
)

large_hole_radius = 35
large_hole_points = []
for i in range(6):
    angle = i * 60
    rad = math.radians(angle)
    x = large_hole_radius * math.cos(rad)
    y = large_hole_radius * math.sin(rad)
    large_hole_points.append((x, y))
result = (
    result.faces(">Z")
    .workplane()
    .pushPoints(large_hole_points)
    .hole(large_hole_dia)
)

for i in range(num_slots):
    result = (
        result.faces(">Z")
        .workplane()
        .transformed(offset=(0, 0, 0), rotate=(0, 0, i * slot_angle_increment))
        .slot2D(slot_length, slot_width, 0)
        .cutThruAll()
    )
