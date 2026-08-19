import cadquery as cq

outer_width = 60.0
outer_height = 40.0
duct_length = 80.0
wall_thickness = 2.0
internal_fillet_radius = 3.0
chamfer_distance = 0.5
tab_thickness = 4.0
tab_length = 30.0
tab_height = 30.0
hole_clearance_dia = 5.8
hole_spacing = 10.0

profile = (
    cq.Workplane("XY")
    .moveTo(0, 0)
    .lineTo(0, outer_height)
    .lineTo(outer_width, outer_height)
    .lineTo(outer_width, outer_height / 2 + 5)
    .lineTo(outer_width - 10, outer_height / 2 + 5)
    .lineTo(outer_width - 10, outer_height / 2 - 5)
    .lineTo(outer_width, outer_height / 2 - 5)
    .lineTo(outer_width, 0)
    .close()
)

result = (
    profile
    .extrude(duct_length)
    .shell(-wall_thickness)
    .edges("<<Z")
    .fillet(internal_fillet_radius)
    .edges("|Z")
    .chamfer(chamfer_distance)
    .faces(">Y")
    .workplane()
    .center(0, 0)
    .rect(tab_length, tab_height)
    .extrude(tab_thickness, combine=True)
    .faces(">Y")
    .workplane()
    .pushPoints([(0, -hole_spacing), (0, 0), (0, hole_spacing)])
    .hole(hole_clearance_dia)
)
