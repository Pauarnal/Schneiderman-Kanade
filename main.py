import sys
import numpy as np
from matplotlib import pyplot
import PIL
import PIL.Image
import time

import kanade
import scan
import augmentation


def load_data(file):
    '''carrega les dades del fitxer file'''
    with open(file) as f:
        i = [[float(x) for x in l.split()] for l in f.readlines()]
    return np.array(i)


def show(img):
    '''mostra una imatge quadrada a partir d'una llista'''
    costat = int(np.sqrt(len(img)))
    pyplot.imshow(img.reshape(costat, costat))
    pyplot.show()


if __name__ == '__main__':
    train = loadq = False
    if len(sys.argv) > 1:
        train = sys.argv[1] == 'train'
        if len(sys.argv) > 2:
            loadq = sys.argv[2] == 'loadq'

    if train:
        print('Carregant les dades...')
        print('    cares')
        faces = load_data('corpus/dfFaces_24x24_norm')
        print('      augmentation')
        print('        new samples')
        faces = np.concatenate((faces, augmentation.load_data('faces')))
        print('        augment samples')
        faces = augmentation.augment_data(faces)
        print('    no-cares')
        notfaces = load_data('corpus/NotFaces_24x24_norm')
        print('      augmentation')
        print('        new samples')
        notfaces = np.concatenate((notfaces,
                                   augmentation.load_data('notfaces')))
        print('        augment samples')
        notfaces = augmentation.augment_data(notfaces)

        print('Entrenant el model...')
        is_face = kanade.train_model(faces, notfaces, umbral=15,
                                     load_quantization=loadq)

        print('Provant el model...')
        cares_ok = sum(1 for x in faces if is_face(x))
        n_cares = len(faces)
        print('    Cares ben classificades: {}% ({}/{})'.format(
                                     cares_ok/n_cares*100, cares_ok, n_cares))

        nocares_ok = sum(1 for x in notfaces if not is_face(x))
        n_nocares = len(notfaces)
        print('    No-cares ben classificades: {}% ({}/{})'.format(
                             nocares_ok/n_nocares*100, nocares_ok, n_nocares))

    else:
        print('Carregant el model...')
        is_face = kanade.load_model(umbral=17)

        print('Reconeixent cares...')
        img = PIL.Image.open(sys.argv[1]).convert('L')
        width, height = img.size
        maxsize = 400
        scale = maxsize / max(width, height) if max(width, height) > maxsize else 1
        img = img.resize(map(int, scale*np.array(img.size)), PIL.Image.ANTIALIAS)

        rects = scan.scan(is_face, img)
        if 'nothumbs' not in sys.argv:
            print('Guardant {} thumbnails...'.format(len(rects)))
            for rect in rects:
                img.crop(rect).resize((24, 24)).save('{}.bmp'.format(time.time()))
                time.sleep(0.05)

        scan.draw_result(img, rects)
        img.show()
        img.save('results/{}.bmp'.format(time.time()))
