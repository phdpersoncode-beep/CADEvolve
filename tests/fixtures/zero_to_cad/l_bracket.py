import cadquery as cq

plate_width = 20.0
leg_long = 80.0
leg_short = 60.0
thickness = 8.0
chamfer_dist = 2.0
hole_clearance = 5.5
cbore_diameter = 8.0
cbore_depth = 2.0
hole_spacing_long = 20.0
hole_spacing_short = 20.0
hole_offset_from_corner = plate_width / 2 + 10.0

long_plate = (
    cq.Workplane("XY")
    .box(plate_width, leg_long, thickness)
    .translate((0, leg_long / 2, thickness / 2))
)
short_plate = (
    cq.Workplane("XY")
    .box(leg_short, plate_width, thickness)
    .translate((-leg_short / 2, leg_long - plate_width / 2, thickness / 2))
)
base = long_plate.union(short_plate)
base = base.edges("|Z and <X and >Y").chamfer(chamfer_dist)

points_long = []
for i in range(3):
    y = hole_offset_from_corner + i * hole_spacing_long
    points_long.append((0, y))

points_short = []
for i in range(2):
    x = -hole_offset_from_corner - i * hole_spacing_short
    points_short.append((x, leg_long - plate_width / 2))

all_hole_points = points_long + points_short
result = (
    base.faces(">Z")
    .workplane()
    .pushPoints(all_hole_points)
    .cboreHole(hole_clearance, cbore_diameter, cbore_depth)
)
