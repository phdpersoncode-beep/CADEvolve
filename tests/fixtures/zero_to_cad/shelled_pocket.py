import cadquery as cq
from types import SimpleNamespace as Measures

measures = Measures(
    outer_width=80.0,
    outer_depth=50.0,
    outer_height=20.0,
    wall_thickness=2.0,
    pocket_width=40.0,
    pocket_depth=30.0,
    pocket_height=8.0,
    chamfer=0.8,
    hole_diameter=4.0,
    hole_spacing=30.0,
    hole_edge_margin=10.0,
)
m = measures

base = (
    cq.Workplane("XY")
    .moveTo(-m.outer_width / 2, -m.outer_depth / 2)
    .lineTo(-m.outer_width / 2, m.outer_depth / 2)
    .threePointArc(
        (0, m.outer_depth / 2 + 8.0),
        (m.outer_width / 2, m.outer_depth / 2),
    )
    .lineTo(m.outer_width / 2, -m.outer_depth / 2)
    .close()
    .extrude(m.outer_height)
)
shelled = base.shell(-m.wall_thickness)
pocket = (
    shelled.faces(">Z")
    .workplane()
    .center(0, 0)
    .rect(m.pocket_width, m.pocket_depth)
    .cutBlind(-m.pocket_height)
)
hole_positions = [(-m.hole_spacing, 0), (0, 0), (m.hole_spacing, 0)]
holes = (
    pocket.faces(">Z")
    .workplane()
    .pushPoints(hole_positions)
    .hole(m.hole_diameter)
)
result = holes.edges(">Z or <Z or |Y or |X").chamfer(m.chamfer)
