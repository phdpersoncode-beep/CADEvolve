import cadquery as cq

# Parameters
base_width = 80.0
base_depth = 60.0
base_thickness = 15.0
boss_radius = 12.0
boss_height = 30.0
slot_width = 8.0
slot_length = 45.0
chamfer_dist = 1.0
mount_hole_radius = 4.0
mount_hole_spacing = 40.0
mount_hole_offset = 20.0
rib_thickness = 4.0

# Base block with mounting holes and outer chamfer
base = (
    cq.Workplane("XY")
    .box(base_width, base_depth, base_thickness, centered=(True, True, False))
    .faces(">Z")
    .workplane()
    .pushPoints([
        (-mount_hole_spacing/2, -mount_hole_offset),
        (mount_hole_spacing/2, -mount_hole_offset),
        (-mount_hole_spacing/2, mount_hole_offset),
        (mount_hole_spacing/2, mount_hole_offset)
    ])
    .hole(mount_hole_radius*2)
    .edges(">Z")
    .chamfer(chamfer_dist)
)

# Central boss created by revolving a triangular profile
boss = (
    cq.Workplane("XZ")
    .polyline([
        (0, 0),
        (boss_radius, boss_height/2),
        (0, boss_height)
    ])
    .close()
    .revolve(360)
)

combined = base.union(boss)

# Slotted opening cut through combined part using cutThruAll
combined = (
    combined
    .workplane(offset=0)
    .center(0, 0)
    .rect(slot_length, slot_width)
    .cutThruAll()
)

# Add top rib for stiffness
rib_width = base_width * 0.8
rib = (
    cq.Workplane("XY")
    .box(rib_width, base_depth, rib_thickness, centered=(True, True, False))
    .translate((0, 0, base_thickness + rib_thickness/2))
)

result = combined.union(rib)

# Chamfer on outer vertical edges for safety
result = result.edges("|Z").chamfer(chamfer_dist)
