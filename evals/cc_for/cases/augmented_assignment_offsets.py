"""Augmented assignment (``+=``, ``*=``) on modelling scalars.

Stresses: ``AugAssign`` is both a read and a write of the same name, which is the
hardest shape for reaching-definition renaming to get right.
"""
import cadquery as cq

column_count = 4
column_side = 8.0
column_height = 24.0
gap = 4.0
deck_thickness = 5.0
deck_margin = 6.0

cursor_x = 0.0
total_span = 0.0
frame = None

for column in range(column_count):
    post = (
        cq.Workplane("XY")
        .box(column_side, column_side, column_height)
        .translate((cursor_x, 0.0, column_height / 2))
    )
    frame = post if frame is None else frame.union(post)
    cursor_x += column_side + gap
    total_span += column_side + gap

total_span -= gap
deck = (
    cq.Workplane("XY")
    .box(total_span + deck_margin, column_side + deck_margin, deck_thickness)
    .translate(
        ((total_span - column_side) / 2, 0.0, column_height + deck_thickness / 2)
    )
)
result = frame.union(deck)
