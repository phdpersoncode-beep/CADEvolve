"""A builder class that stores a workplane, mutates it, then reads it back.

Stresses: copy propagation across an opaque method call.  ``self.wp`` is rebound
by ``build()``, so replacing the later read with the constructor argument it was
initialised from silently yields an empty part -- with no exception and no
structural error, which is the failure mode only a solid-level comparison finds.
"""
import cadquery as cq


class VentedCover:
    def __init__(self, workplane, plate_x, plate_y, plate_z, vent_diameter, vent_pitch):
        self.wp = workplane
        self.plate_x = plate_x
        self.plate_y = plate_y
        self.plate_z = plate_z
        self.vent_diameter = vent_diameter
        self.vent_pitch = vent_pitch
        self.build()
        self.model = self.wp

    def build(self):
        self.wp = self.wp.box(self.plate_x, self.plate_y, self.plate_z)
        self.wp = (
            self.wp.faces(">Z")
            .workplane()
            .rarray(self.vent_pitch, self.vent_pitch, 3, 2)
            .hole(self.vent_diameter)
        )


result = VentedCover(cq.Workplane("XY"), 60.0, 40.0, 6.0, 5.0, 14.0).model
