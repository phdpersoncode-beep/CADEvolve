import cadquery as cq
cap_outer_diameter = 60.0
wall_thickness = 2.0
cap_height = 20.0
groove_depth = 0.5
groove_height = 1.5
vent_slot_width = 8.0
vent_slot_height = 4.0
vent_slot_depth = wall_thickness - 0.2
vent_offset_from_top = 3.0
chamfer_distance = 0.5
inner_radius = (cap_outer_diameter / 2) - wall_thickness
outer_radius = cap_outer_diameter / 2
outer = cq.Workplane('XY').circle(outer_radius).extrude(cap_height)
inner = cq.Workplane('XY').circle(inner_radius).extrude(cap_height)
body = outer.cut(inner)
groove_cyl = cq.Workplane('XY').circle(inner_radius - groove_depth).extrude(groove_height).translate((0,0,cap_height - groove_height))
body = body.cut(groove_cyl)
vent_center_z = cap_height - vent_offset_from_top - vent_slot_height/2
vent_cut = (
    cq.Workplane('XY')
    .box(vent_slot_depth, vent_slot_width, vent_slot_height)
    .translate((outer_radius - vent_slot_depth/2, 0, vent_center_z))
)
body = body.cut(vent_cut)
body = body.edges('|Z').chamfer(chamfer_distance)
result = body
