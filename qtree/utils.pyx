import numpy as np
cimport numpy as np

def _points_in_poly(np.float64_t[:, :] verts, np.float64_t[:, :] px,
                    np.float64_t[:, :] py):
    # see https://stackoverflow.com/a/2922778/1382869
    cdef np.intp_t[:, :] c
    cdef np.intp_t  i, j, k

    c = np.zeros((px.shape[0], px.shape[1]), dtype='int')
    for i in range(verts.shape[0]):
        for j in range(px.shape[0]):
            for k in range(px.shape[1]):
                if (((verts[i, 1] > py[j, k]) != (verts[i-1, 1] > py[j, k]))
                    and (px[j, k] < (
                        (verts[i-1, 0] - verts[i, 0]) *
                        (py[j, k] - verts[i, 1]) /
                        (verts[i-1, 1] - verts[i, 1])) + verts[i, 0])):
                    c[j, k] += 1
    return np.asarray(c) % 2
