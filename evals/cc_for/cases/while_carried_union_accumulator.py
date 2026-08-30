"""A ``while`` that unions each pass into an accumulator seeded with ``None``.

Stresses: the loop-carried geometry accumulator, in the loop form the ``for``
cases do not reach.  ``pegs`` is written by a conditional expression whose two
branches are both CadQuery values, so lowering rewrites the expression into an
``if``/``else`` over a ``wpN``, and the assignment back to ``pegs`` is what makes
the next iteration -- and the union after the loop -- see anything at all.

This is the shape that fails silently when it is missed: the canonical program
still runs, still passes the structural contract, and simply builds the base
plate with no pegs on it.
"""
import cadquery as cq

plate_size = 44.0
plate_thickness = 4.0
peg_count = 3
peg_pitch = 11.0
peg_radius = 2.5
peg_height = 6.0

base = cq.Workplane("XY").box(plate_size, plate_size, plate_thickness)

pegs = None
index = 0
while index < peg_count:
    peg = (
        cq.Workplane("XY")
        .center(index * peg_pitch - peg_pitch, 0)
        .circle(peg_radius)
        .extrude(peg_height)
    )
    pegs = peg if pegs is None else pegs.union(peg)
    index = index + 1

result = base.union(pegs)
