import numpy

# PCA


def estimate_pca(data, d, model):
    '''Estima PCA per a data (per files) i d dimensions.
    Guarda el resultat en model.'''
    print("        estimant PCA...")
    mu = sum(data)/len(data)
    data_prep = data - mu

    sigma = numpy.cov(data_prep.T)
    u, s, v = numpy.linalg.svd(sigma)
    ureduceT = u[:, :d].T

    model['mu'], model['ureduceT'] = mu, ureduceT
    return lambda x: ureduceT.dot(x - mu)


def load_pca(model):
    '''carrega un PCA estimat des d'un fitxer'''
    mu, ureduceT = model['mu'], model['ureduceT']
    return lambda x: ureduceT.dot(x - mu)

# K-mitjanes


def nearest(x, centroids):
    '''retorna el centroide mes proper'''
    return numpy.argmin(numpy.linalg.norm(x - centroids, axis=1))


def estimate_kmeans(data, k, n, model):
    '''Estima k-means per a data (per files) amb k centroides i n iteracions.
    Guarda el resultat en model.'''
    print("        estimant K-means...")

    # inicialitzem aleatoriament
    centroids = numpy.random.permutation(data)[:k]

    # afegim un 1 a la ultima columna de les dades per a acumular
    data_ac = numpy.ones((len(data), len(data[0])+1))
    data_ac[:, :-1] = data

    for i in range(n):
        print("            iteració {}".format(i+1))
        acc = numpy.zeros((k, len(data_ac[0])))
        # acumulació
        for vec in data_ac:
            acc[nearest(vec[:-1], centroids)] += vec
        # mitjana
        for vec in acc:
            vec /= vec[-1]
        centroids = acc[:, :-1]

    model['centroids'] = centroids

    return lambda x: nearest(x, centroids)


def load_kmeans(model):
    '''carrega un K-means estimat des de model'''
    centroids = model['centroids']
    return lambda x: nearest(x, centroids)

# Quantització (PCA + K-mitjanes)


def estimate_quantization(data, k, n, d, model):
    '''estima la classificacio dels patrons amb PCA i k-means.
    Guarda el resultat en model.'''
    pca = estimate_pca(data, d, model)
    kmeans = estimate_kmeans(numpy.array([pca(v) for v in data]), k, n, model)
    return lambda x: kmeans(pca(x))


def load(model):
    '''carrega una quantització estimada des de model.'''
    pca = load_pca(model)
    kmeans = load_kmeans(model)
    return lambda x: kmeans(pca(x))

# Entrenament de la quantització a partir de les dades


def patterns_from_image(img, n=6, m=24):
    '''extrau els patrons nxn de la imatge de mxm'''
    img = img.reshape(m, m)
    pats = [img[n*i:n*(i+1), n*j:n*(j+1)].flatten() for j in range(m//n)
                                                    for i in range(m//n)]
    return numpy.array(pats)


def patterns_from_data(data):
    '''extrau els patrons de les imatges de data'''
    imgs_w_pats = (patterns_from_image(img) for img in data)
    return numpy.array([pat for img in imgs_w_pats for pat in img])


def train(data, model, k=100, n=10, d=8):
    '''entrena la quantització a partir de les dades. Guarda en model.'''
    return estimate_quantization(patterns_from_data(data), k, n, d, model)

# Quantització de les dades amb l'entrenament ja fet


def quantize_image(img, quantize):
    '''quantitza les subregions de img amb la funció quantize'''
    return numpy.array([quantize(pat) for pat in patterns_from_image(img)])


def quantize_data(data, quantize):
    '''quantitza les dades amb la funció quantize'''
    return numpy.array([quantize_image(img, quantize) for img in data])
