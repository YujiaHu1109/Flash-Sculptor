import numpy as np
import trimesh

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def cartesian_to_spherical(x, y, z):
    """
    将笛卡尔坐标转换为球坐标 (phi, theta)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)  # 极角
    phi = np.arctan2(y, x)  # 方位角
    return phi, theta


def get_icosphere_spherical_coords(sub=2):
    """
    获取 icosphere 的顶点对应的球坐标 (phi, theta)
    """
    # 创建细分级别为 2 的 icosphere，其顶点数为 42
    icosphere = trimesh.creation.icosphere(subdivisions=sub)
    vertices = icosphere.vertices

    spherical_coords = []
    for vertex in vertices:
        x, y, z = vertex
        phi, theta = cartesian_to_spherical(x, y, z)
        spherical_coords.append([phi, theta])

    return spherical_coords