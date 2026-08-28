"""A parameter list mutated in place across loop iterations.

Stresses: ``offsets`` is bound once but its *contents* change every pass, so
hoisting the binding into the preamble is only safe if the mutation stays inside
the loop, in order.
"""
import cadquery as cq

rib_count = 4
rib_thickness = 3.0
rib_height = 9.0
base_length = 64.0
base_width = 26.0
base_thickness = 5.0
growth = 1.35

offsets = [-base_length / 2 + 8.0]
widths = [rib_thickness]

for step in range(1, rib_count):
    offsets.append(offsets[-1] + 12.0)
    widths.append(widths[-1] * growth)

base = cq.Workplane("XY").box(base_length, base_width, base_thickness)

for offset, width in zip(offsets, widths):
    base = (
        base.faces(">Z")
        .workplane()
        .center(offset, 0.0)
        .rect(width, base_width)
        .extrude(rib_height)
    )

result = base
