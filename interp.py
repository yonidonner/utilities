import itertools

import numpy
import scipy.sparse

def make_interpolation_matrix(input_shape, C):
    shp = C.shape
    d = shp[-1]
    shp = shp[:-1]
    assert len(input_shape) == d
    C1 = C.reshape((-1, d))
    N = C1.shape[0]
    I = numpy.floor(C1)
    R = C1 - I
    I = I.astype(numpy.int32)
    f = lambda i: [
        numpy.clip(I[:, i] + j, 0, input_shape[i]-1) for j in [0, 1]]
    Is = [f(i) for i in range(d)]
    big_I = numpy.stack(
        [numpy.stack(x, 1) for x in itertools.product(*Is)], 1).reshape((-1, d))
    S = numpy.multiply.accumulate([1]+list(reversed(input_shape))[:-1])[::-1]
    big_I = big_I.dot(S)
    Rs = [numpy.stack([1 - R[:, i], R[:, i]], 1) for i in range(d)]
    for i in range(d):
        sh = [N] + [1] * d
        sh[i+1] = 2
        R_i = Rs[i].reshape(sh)
        if i == 0:
            big_R = R_i
        else:
            big_R = big_R * R_i
    big_R = big_R.reshape((-1,))
    indptr = numpy.arange(N+1, dtype=numpy.int32) * 2**d
    return (big_R, big_I, indptr), (shp, input_shape)

def make_applier(matrix_params):
    (big_R, big_I, indptr), (shp, input_shape) = matrix_params
    N1 = numpy.prod(shp)
    d = len(input_shape)
    N2 = numpy.prod(input_shape)
    M = scipy.sparse.csr_matrix(
        (big_R, big_I, indptr), shape=(N1, N2), dtype=big_R.dtype)
    def apply_matrix(X):
        sh = list(X.shape[d:])
        return M.dot(X.reshape((-1, numpy.prod(sh)))).reshape(list(shp) + sh)
    return apply_matrix

def test_interp(image_filename, out_shape, dtype=numpy.float64):
    import time
    mm = lambda x, M: x.reshape((-1, M.shape[0])).dot(M).reshape(x.shape)
    import imageio
    im = imageio.imread(image_filename)
    C1 = numpy.meshgrid(
        *[numpy.arange(out_shape[i], dtype=dtype)
          for i in xrange(len(out_shape))], indexing='ij')
    C = numpy.stack(C1, 2) - 0.5 * (numpy.array(out_shape) - 1.0)
    shift = 0.5 * (numpy.array(im.shape[:2]) - 1.0)
    C1 = []
    n1 = 10
    for i in xrange(n1):
        th = numpy.pi * 2 * i / n1
        cth = numpy.cos(th)
        sth = numpy.sin(th)
        M = numpy.array([[cth, sth], [-sth, cth]])
        C1.append(mm(C, M) + shift)
    C1 = numpy.stack(C1, 0)
    t0 = time.time()
    matrix_params = make_interpolation_matrix(im.shape[:2], C1)
    t1 = time.time()
    f_apply = make_applier(matrix_params)
    t2 = time.time()
    print t1-t0, t2-t1
    for i in xrange(10):
        t0 = time.time()
        y1 = f_apply(im)
        t1 = time.time()
        print t1-t0
    for i in xrange(y1.shape[0]):
        im = numpy.clip(y1[i, ...], 0, 255).astype(numpy.uint8)
        imageio.imwrite('test%03d.png'%(i,), im)
