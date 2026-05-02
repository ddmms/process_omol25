import hashlib
import time


def geom_sha1_old(elems, coords, ndp: int = 6):
    h = hashlib.sha1()
    for e, (x, y, z) in zip(elems, coords):
        h.update(
            f"{e}:{round(x, ndp):.6f}:{round(y, ndp):.6f}:{round(z, ndp):.6f};".encode()
        )
    return h.hexdigest()


def geom_sha1_new(elems, coords, ndp: int = 6):
    # Using generator expression as suggested by the instructions
    return hashlib.sha1(
        "".join(
            f"{e}:{round(x, ndp):.6f}:{round(y, ndp):.6f}:{round(z, ndp):.6f};"
            for e, (x, y, z) in zip(elems, coords)
        ).encode()
    ).hexdigest()


def geom_sha1_listcomp(elems, coords, ndp: int = 6):
    return hashlib.sha1(
        "".join(
            [
                f"{e}:{round(x, ndp):.6f}:{round(y, ndp):.6f}:{round(z, ndp):.6f};"
                for e, (x, y, z) in zip(elems, coords)
            ]
        ).encode()
    ).hexdigest()


elems = ["C", "H", "O"] * 30
coords = [(1.123456, 2.123456, 3.123456)] * 90

N = 10000

t0 = time.time()
for _ in range(N):
    geom_sha1_old(elems, coords)
print("Old:", time.time() - t0)

t0 = time.time()
for _ in range(N):
    geom_sha1_new(elems, coords)
print("New gen:", time.time() - t0)

t0 = time.time()
for _ in range(N):
    geom_sha1_listcomp(elems, coords)
print("New listcomp:", time.time() - t0)
