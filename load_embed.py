import numpy as np

def load_embedding(category):
    if category == 'Laptops':
        return np.load('K:/recommendation system/dataset/Laptops.npy')
    elif category == 'smartphone':
        return np.load('K:/recommendation system/dataset/smartphone.npy')
    elif category == 'Basic Cases':
        return np.load('K:/recommendation system/dataset/Basic Cases.npy')
    elif category == 'Headphone':
        return np.load('K:/recommendation system/dataset/Headphone.npy')
    elif category == 'Laptop Bags':
        return np.load('K:/recommendation system/dataset/Laptop Bags.npy')
    elif category == 'Screen Protector':
        return np.load('K:/recommendation system/dataset/Screen Protector.npy')
    elif category == 'Phone Charger':
        return np.load('K:/recommendation system/dataset/Phone Charger.npy')
    elif category == 'mouse':
        return np.load('K:/recommendation system/dataset/mouse.npy')
    elif category == 'Laptop Charger':
        return np.load('K:/recommendation system/dataset/Laptop Charger.npy')


