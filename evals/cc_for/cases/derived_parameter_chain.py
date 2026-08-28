"""A long chain of derived parameters ending in a trigonometric work plane.

Stresses: the property the whole change exists for.  Every dimension below is a
symbolic function of ``mount_angle_deg``; if canonicalization freezes any link the
perturbation gate diverges even though the unperturbed solids match exactly.
"""
import cadquery as cq
import math

mount_angle_deg = 35.0
arm_length = 46.0
arm_width = 14.0
arm_thickness = 8.0
pad_thickness = 5.0
bolt_diameter = 6.0

mount_angle_rad = math.radians(mount_angle_deg)
lever = arm_length * math.cos(mount_angle_rad)
rise = arm_length * math.sin(mount_angle_rad)
pad_length = lever * 0.45
pad_width = arm_width * 1.2
normal_vector = (math.sin(mount_angle_rad), 0.0, math.cos(mount_angle_rad))
x_direction = (math.cos(mount_angle_rad), 0.0, -math.sin(mount_angle_rad))
mount_plane = cq.Plane(
    origin=(lever / 2, 0.0, rise / 2), xDir=x_direction, normal=normal_vector
)

arm = cq.Workplane("XY").box(arm_length, arm_width, arm_thickness)
pad = cq.Workplane(mount_plane).rect(pad_length, pad_width).extrude(pad_thickness)
joined = arm.union(pad)
result = (
    joined.faces("<Z")
    .workplane()
    .center(-arm_length / 2 + bolt_diameter, 0.0)
    .circle(bolt_diameter / 2)
    .cutThruAll()
)
