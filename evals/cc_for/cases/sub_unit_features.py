"""Features smaller than one unit, which integer binarization has to round.

Stresses: the quantized gate.  Sub-millimetre fillets, walls and grooves are the
case where the source and the binarized part legitimately differ, so IoU and
Chamfer are expected to be close but not equal.
"""
import cadquery as cq

shell_length = 30.0
shell_width = 20.0
shell_height = 10.0
wall_thickness = 0.8
edge_fillet = 0.4
groove_width = 0.6
groove_depth = 0.35
vent_diameter = 0.9
vent_pitch = 5.0
vent_count = 4

blank = cq.Workplane("XY").box(shell_length, shell_width, shell_height)
rounded = blank.edges("|Z").fillet(edge_fillet)
shell = rounded.faces(">Z").shell(-wall_thickness)
grooved = (
    shell.faces(">X")
    .workplane(centerOption="CenterOfMass")
    .rect(groove_width, shell_width * 0.6)
    .cutBlind(-groove_depth)
)
vent_points = [((i - (vent_count - 1) / 2) * vent_pitch, 0.0) for i in range(vent_count)]
result = (
    grooved.faces("<Z")
    .workplane()
    .pushPoints(vent_points)
    .hole(vent_diameter)
)
