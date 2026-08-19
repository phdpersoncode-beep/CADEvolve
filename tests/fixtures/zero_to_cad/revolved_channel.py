import cadquery as cq

outer_radius = 40
wall_thickness = 8
channel_height = 30
fillet_radius = 5
hole_diameter = 6
chamfer_size = 1.5
tab_width = 20
tab_height = 10
tab_spacing_fraction = 0.5
tab_start1 = channel_height * 0.2
tab_start2 = channel_height * 0.6
tab_end1 = tab_start1 + tab_height
tab_end2 = tab_start2 + tab_height
rib_width = 4
rib_height = 12
rib_depth = wall_thickness * 0.6
rib_positions = [
    channel_height * 0.25,
    channel_height * 0.5,
    channel_height * 0.75,
]
inner_radius = outer_radius - wall_thickness

profile = (
    cq.Workplane("XZ")
    .moveTo(outer_radius, 0)
    .lineTo(outer_radius, tab_start1)
    .lineTo(outer_radius + tab_width, tab_start1)
    .lineTo(outer_radius + tab_width, tab_end1)
    .lineTo(outer_radius, tab_end1)
    .lineTo(outer_radius, tab_start2)
    .lineTo(outer_radius + tab_width, tab_start2)
    .lineTo(outer_radius + tab_width, tab_end2)
    .lineTo(outer_radius, tab_end2)
    .lineTo(outer_radius, channel_height)
    .lineTo(inner_radius + fillet_radius, channel_height)
    .threePointArc(
        (inner_radius, channel_height),
        (inner_radius, channel_height - fillet_radius),
    )
    .lineTo(inner_radius, 0)
    .close()
)
c_channel = profile.revolve(180)
c_channel = c_channel.edges("|Z and >X").chamfer(chamfer_size)
c_channel = c_channel.faces(">X").hole(hole_diameter)
c_channel = (
    c_channel.faces("<X")
    .workplane()
    .pushPoints([(0, y) for y in rib_positions])
    .rect(rib_width, rib_height)
    .cutBlind(rib_depth)
)
result = c_channel
