import cadquery as cq
import math

S = 21
C = 5
H = 16
K = 3
A = 0.9
M = 17

wp = cq.Workplane("XZ")
wp = (
    wp.moveTo(0, S)
    .threePointArc((S * math.sqrt(2) / 2, S * math.sqrt(2) / 2), (S, 0))
    .lineTo(0, 0)
    .close()
)
body = wp.revolve(360, axisStart=(0, 0, 0), axisEnd=(0, 1, 0))
obj = body
for i in range(K):
    frac = (i + 1) / (K + 1)
    z = -H / 2 + frac * H
    r = math.sqrt(max(0, S * S - z * z))
    h = A * r
    top = (0.75 * r, z + h / 2)
    bot = (0.25 * r, z - h / 2)
    for j in range(M):
        th = 2 * math.pi * j / M
        x = r * math.cos(th)
        y = r * math.sin(th)
        pl = cq.Plane(origin=(x, y, z), normal=(x / S, y / S, z / S))
        bump = cq.Workplane(pl).circle(A).extrude(h)
        obj = obj.union(bump)
result = obj
