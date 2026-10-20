import struct
import math

MODEL = "model/stories260K.bin"

DIM = 64
CONFIG_SIZE = 7 * 4

# From the verified checkpoint layout
RMS_ATT_OFFSET = 131100

EPS = 1e-5


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


with open(MODEL, "rb") as f:

    # ---------------------------------------------------------
    # Read first 64 embedding values
    # ---------------------------------------------------------

    f.seek(CONFIG_SIZE)

    x = [read_f32(f) for _ in range(DIM)]

    # ---------------------------------------------------------
    # Read layer-0 attention RMSNorm weights
    # ---------------------------------------------------------

    f.seek(RMS_ATT_OFFSET)

    w = [read_f32(f) for _ in range(DIM)]


# -------------------------------------------------------------
# RMSNorm
# -------------------------------------------------------------

ss = sum(v * v for v in x) / DIM

rms = math.sqrt(ss + EPS)

y = [(x[i] / rms) * w[i] for i in range(DIM)]


print("RMSNorm PC Reference")
print("====================")

print(f"RMS value = {rms:.12f}")
print()

for i in range(8):
    print(
        f"x[{i:2d}] = {x[i]: .12f}   "
        f"w[{i:2d}] = {w[i]: .12f}   "
        f"y[{i:2d}] = {y[i]: .12f}"
    )