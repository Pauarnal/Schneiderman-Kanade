import numpy
import quantization


def train_faces(data, k, model):
    '''Entrena el model de cares. Guarda el resultat en un model.'''
    p = numpy.ones((k, len(data[0])))  # suavitzat pla
    for img in data:
        for i in range(len(img)):
            p[img[i], i] += 1
    p /= len(data)
    p = numpy.log(p)
    model['faces'] = p
    return lambda img: numpy.sum([p[img[i], i] for i in range(len(img))])


def load_faces(model):
    '''Carrega el model de cares des de model.'''
    p = model['faces']
    return lambda img: numpy.sum([p[img[i], i] for i in range(len(img))])


def train_notfaces(data, k, model):
    '''Entrena el model de no-cares. Guarda el resultat en model.'''
    p = numpy.ones(k)  # suavitzat pla
    for img in data:
        for i in range(len(img)):
            p[img[i]] += 1
    p /= len(data) * len(data[0])
    p = numpy.log(p)
    model['notfaces'] = p
    return lambda img: numpy.sum(p[img[i]] for i in range(len(img)))


def load_notfaces(model):
    '''Carrega el model de no-cares des de model.'''
    p = model['notfaces']
    return lambda img: numpy.sum(p[img[i]] for i in range(len(img)))


def train_quant_model(faces, notfaces, umbral, k, model):
    '''Entrena Kanade per a img quantitzada. Guarda en model.'''
    p_faces = train_faces(faces, k, model)
    p_notfaces = train_notfaces(notfaces, k, model)
    return lambda img: p_faces(img) - p_notfaces(img) >= umbral


def load_quant_model(model, umbral):
    '''Carrega Schneiderman-Kanade per a img quantitzada des de model'''
    p_faces = load_faces(model)
    p_notfaces = load_notfaces(model)
    return lambda img: p_faces(img) - p_notfaces(img) >= umbral


def train_model(faces, notfaces, umbral=20, load_quantization=False,
                file_kanade='model/kanade.npz',
                file_quantiz='model/quantization.npz'):
    '''Entrena el model de Schneiderman-Kanade. Guarda en fitxers.
    Opcionalment, carrega una quantització entrenada.'''
    if load_quantization:
        print('    carregant quantització...')
        model_quantiz = numpy.load(file_quantiz)
        quantize = quantization.load(model_quantiz)
    else:
        print('    entrenant quantització...')
        model_quantiz = {}
        quantize = quantization.train(numpy.concatenate((faces, notfaces)),
                                      model_quantiz)
        numpy.savez(file_quantiz, **model_quantiz)

    print("    quantitzant les dades d'entrenament...")
    print('        cares')
    faces_q = quantization.quantize_data(faces, quantize)
    print('        no-cares')
    notfaces_q = quantization.quantize_data(notfaces, quantize)

    print('    entrenant el model Schneiderman-Kanade...')
    model_kanade = {}
    k = len(model_quantiz['centroids'])  # número de centroides
    isface_q = train_quant_model(faces_q, notfaces_q, umbral, k, model_kanade)
    numpy.savez(file_kanade, **model_kanade)
    return lambda img: isface_q(quantization.quantize_image(img, quantize))


def load_model(umbral=20, file_kanade='model/kanade.npz',
               file_quantiz='model/quantization.npz'):
    '''Carrega el model de Schneiderman-Kanade des de fitxers.'''
    print('    quantització')
    model_quantiz = numpy.load(file_quantiz)
    quantize = quantization.load(model_quantiz)
    print('    schneiderman-kanade')
    model_kanade = numpy.load(file_kanade)
    isface_q = load_quant_model(model_kanade, umbral)
    return lambda img: isface_q(quantization.quantize_image(img, quantize))
