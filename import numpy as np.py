import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os


base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extrair_caracteristicas(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Extrai as características usando o modelo VGG16
    features = model.predict(x)
    return features

def encontrar_similares(imagem_consulta, pasta_imagens, n_recomendacoes=5):
    # Extrai características da imagem de consulta
    features_consulta = extrair_caracteristicas(imagem_consulta)
    
    # Lista para armazenar características e caminhos das imagens
    todas_features = []
    caminhos_imagens = []
    
 
    for img_name in os.listdir(pasta_imagens):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(pasta_imagens, img_name)
            features = extrair_caracteristicas(img_path)
            todas_features.append(features.flatten())
            caminhos_imagens.append(img_path)
    
    # Calcula similaridade entre a imagem de consulta e todas as outras
    todas_features = np.array(todas_features)
    similaridades = cosine_similarity(features_consulta, todas_features)
    
    # Encontra os índices das imagens mais similares
    indices_similares = similaridades.argsort()[0][-n_recomendacoes:][::-1]
    
    return [caminhos_imagens[i] for i in indices_similares]

def mostrar_recomendacoes(imagem_consulta, imagens_similares):

    plt.figure(figsize=(15, 3))
    
    
    plt.subplot(1, 6, 1)
    plt.imshow(image.load_img(imagem_consulta))
    plt.title('Consulta')
    plt.axis('off')
    
    # Mostra as imagens recomendadas
    for i, img_path in enumerate(imagens_similares):
        plt.subplot(1, 6, i + 2)
        plt.imshow(image.load_img(img_path))
        plt.title(f'Rec {i+1}')
        plt.axis('off')
    
    plt.show()

