import struct

MODEL = "model/stories260K.bin"

DIM = 64
CONFIG_SIZE = 7 * 4

# Verified checkpoint offsets
EMBEDDING_OFFSET = 28
RMS_ATT_OFFSET = 131100
WQ_OFFSET = 132380


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


# ---------------------------------------------------------
# Load first embedding vector
# ---------------------------------------------------------

with open(MODEL, "rb") as f:

    f.seek(EMBEDDING_OFFSET)

    x = [read_f32(f) for _ in range(DIM)]

    # -----------------------------------------------------
    # Load layer-0 attention RMSNorm weights
    # -----------------------------------------------------

    f.seek(RMS_ATT_OFFSET)

    rms_weight = [read_f32(f) for _ in range(DIM)]

    # -----------------------------------------------------
    # Load layer-0 WQ
    #
    # WQ shape:
    #     [64 x 64]
    # -----------------------------------------------------

    f.seek(WQ_OFFSET)

    wq = [
        [read_f32(f) for _ in range(DIM)]
        for _ in range(DIM)
    ]


# ---------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------

EPS = 1e-5

sum_sq = sum(v * v for v in x)

rms = (sum_sq / DIM + EPS) ** 0.5

norm_x = [
    x[i] / rms * rms_weight[i]
    for i in range(DIM)
]


# ---------------------------------------------------------
# Matrix multiplication
#
# q[row] = sum(W[row][col] * norm_x[col])
# ---------------------------------------------------------

q = []

for row in range(DIM):

    value = 0.0

    for col in range(DIM):
        value += wq[row][col] * norm_x[col]

    q.append(value)


# ---------------------------------------------------------
# Print golden reference
# ---------------------------------------------------------

print("TinyStories 260K WQ MatMul Reference")
print("====================================")

print(f"Input dimension  : {DIM}")
print(f"Output dimension : {DIM}")
print(f"WQ offset        : {WQ_OFFSET}")
print()

print("Q vector:")
print()

for i in range(16):
    print(
        f"q[{i:2d}] = {q[i]: .12f}"
    )