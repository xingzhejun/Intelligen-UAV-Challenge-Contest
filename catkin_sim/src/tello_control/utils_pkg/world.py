# -*- coding: utf-8 -*-
import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def nvec3(x, y, z):
    v = vec3(x, y, z)
    return v/np.linalg.norm(v)


def nm(v):
    return v/np.linalg.norm(v)


def vec2hpr(v):
    return vec3(np.arctan2(-v[0], v[1]), np.arctan2(v[2], np.linalg.norm(v[0:-1])), 0)


def hpr2vec(hpr):
    return nvec3(np.tan(-hpr[0]), 1, np.tan(hpr[1]) / np.cos(hpr[0]))


def hpr_add(v1, v2):
    pass


def hpr2matrix(t):
    ca = np.cos(t[2])
    sa = np.sin(t[2])
    cb = np.cos(t[1])
    sb = np.sin(t[1])
    cg = np.cos(t[0])
    sg = np.sin(t[0])
    return np.array([
        [ca*cg-cb*sa*sg, -cb*cg*sa-ca*sg, sa*sb],
        [cg*sa+ca*cb*sg, ca*cb*cg-sa*sg, -ca*sb],
        [sb*sg, cg*sb, cb]
    ], dtype=np.float64)






