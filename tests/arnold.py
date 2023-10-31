from numpy import asarray, eye, indices
from numpy.linalg import matrix_power
from numpy.typing import ArrayLike


def arnold_transform1(img: ArrayLike, k: int) -> ArrayLike:
    T = eye(2, dtype=int)  # 2x2 Birim matris

    for _ in range(2 * k):
        T[:] = [[T[1, 0], T[1, 1]], [T[1, 1], T[1, 0] + T[1, 1]]]

    T_inv = asarray([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]])

    return coordinate_transform(img, T), T_inv


def arnold_transform(image, k, p=1, q=1):
    T = matrix_power([[1, p], [q, p * q + 1]], k)

    T_inv = asarray(
        [
            [T[1, 1], -T[0, 1]],
            [-T[1, 0], T[0, 0]],
        ]
    )

    return coordinate_transform(image, T), T_inv


def coordinate_transform(image, T):
    i = indices(image.shape)
    i = (T @ i.reshape(2, -1)).reshape(i.shape).astype(int)

    return image[i[0] % image.shape[0], i[1] % image.shape[1]]
