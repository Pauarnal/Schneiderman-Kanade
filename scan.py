import numpy
import PIL
import PIL.Image
import PIL.ImageDraw


def integ_use(integ_img, indices):
    i, j, n = indices
    return (integ_img[i+n, j+n] - integ_img[i, j+n] -
            integ_img[i+n, j] + integ_img[i, j]) / (n*n)


def normalize(img, integ_img, indices):
    i, j, n = indices
    integ_1, integ_2 = integ_img

    mean = integ_use(integ_1, indices)
    std = numpy.sqrt(integ_use(integ_2, indices) - mean*mean)

    return (img[i:i+n, j:j+n] - mean) / std


def integral(img):
    integ_img = numpy.zeros((lambda x, y: (x+1, y+1))(*img.shape))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            integ_img[i+1, j+1] = (integ_img[i+1, j] + integ_img[i, j+1] +
                                   img[i, j] - integ_img[i, j])

    return integ_img


def scan_basic(is_face, img, n=24):
    '''torna els indexs on ha trobat cara en una imatge de grandaria fixa.'''
    print('    scanning...')
    img = numpy.array(img, copy=False, dtype=int)

    integ_1 = integral(img)
    integ_2 = integral(numpy.square(img))

    return ((i, j) for i in range(len(img)-n+1)
                   for j in range(len(img[0])-n+1)
                   if is_face(normalize(img, (integ_1, integ_2), (i, j, n))))


def rescale(img, scale):
    print('    rescale to {}'.format(scale))
    return img.resize(map(int, scale*numpy.array(img.size)),
                      PIL.Image.ANTIALIAS)


def scan_faces_scale(is_face, img, scale):
    idxs = scan_basic(is_face, rescale(img, scale))
    rects = ((x//scale, y//scale,
              x//scale + 24//scale, y//scale + 24//scale) for (x, y) in idxs)
    return rects


def scan(is_face, img, min=24):
    list_lfaces = (scan_faces_scale(is_face, img, s)
                   for s in numpy.geomspace(0.1, 1.0, 12))
    return [(y, x, y2, x2) for faces in list_lfaces
                           for (x, y, x2, y2) in faces]


def draw_result(img, rects):
    draw = PIL.ImageDraw.Draw(img)
    for rect in rects:
        draw.rectangle(list(rect), outline=255)
