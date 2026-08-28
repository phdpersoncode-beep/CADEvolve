import cadquery as cq
from types import SimpleNamespace as Measures

params = Measures(
    bar_length=80.0,
    bar_width=30.0,
    bar_height=10.0,
    corner_fillet=2.0,
    cbore_position=20.0,
    cbore_diameter=4.0,
    cbore_counterbore_diameter=7.0,
    cbore_depth=5.0,
    notch_width=12.0,
    notch_height=6.0,
    notch_offset=15.0,
    notch_depth=12.0,
)
p = params

profile = (
    cq.Workplane("XY")
    .moveTo(0, 0)
    .lineTo(p.bar_length, 0)
    .lineTo(p.bar_length, p.bar_width)
    .lineTo(p.corner_fillet, p.bar_width)
    .threePointArc((0, p.bar_width - p.corner_fillet), (0, 0))
    .close()
)
bar = profile.extrude(p.bar_height)
bar = (
    bar.faces(">Y")
    .workplane()
    .center(p.notch_offset, p.bar_height / 2)
    .rect(p.notch_width, p.notch_height)
    .cutBlind(-p.notch_depth)
)
bar = (
    bar.faces(">Z")
    .workplane()
    .pushPoints([(p.cbore_position, p.bar_width / 2)])
    .cboreHole(
        p.cbore_diameter,
        p.cbore_counterbore_diameter,
        p.cbore_depth,
    )
)
bar = bar.faces(">Z").edges().fillet(p.corner_fillet)
result = bar
