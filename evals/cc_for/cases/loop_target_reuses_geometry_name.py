"""Two loops in one function scope, the second iterating over the first's name.

Stresses: inside a ``def`` the reaching-definition renamer leaves names alone --
they are that scope's locals -- so ``stud`` is both the first loop's assignment
target and the second loop's iteration variable.  Lowering aliases the first to
its ``wpN``; unless the second loop's target clears that alias, every pass of the
union reads the last stud the first loop built instead of the one it is on.

The canonical program still runs and still keeps the structural contract; it just
unions one stud three times, so the part comes out with two studs missing.
"""
import cadquery as cq

plate_size = 40.0
plate_thickness = 4.0
stud_size = 4.0
stud_height = 6.0
stud_pitch = 12.0
stud_count = 3


def build_plate():
    plate = cq.Workplane("XY").box(plate_size, plate_size, plate_thickness)
    studs = []
    for index in range(stud_count):
        stud = (
            cq.Workplane("XY")
            .box(stud_size, stud_size, stud_height)
            .translate((index * stud_pitch - stud_pitch, 0, plate_thickness / 2))
        )
        studs.append(stud)
    for stud in studs:
        plate = plate.union(stud)
    return plate


result = build_plate()
