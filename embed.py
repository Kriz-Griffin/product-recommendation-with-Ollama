from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('dataset/boston data 1.csv')

def generate_embeddings(dataset, category, save_path):
    """Generate and save embeddings for a specific product category."""
    filtered_data = dataset[dataset['category'] == category]

    # Combine relevant features into a single text representation
    filtered_data['combined_features'] = (
       
        filtered_data['product_title'].astype(str) + " " +
        ((filtered_data['brand'] + " ") * 20).astype(str) + " " +
        ((filtered_data['model'] + " ") * 5).astype(str) + " " +
        filtered_data['features_preporcessed'].astype(str)
    )

    # Load model and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(filtered_data['combined_features'].tolist(), convert_to_numpy=True)

    # Save embeddings to file
    np.save(save_path, embeddings)

# Define categories and corresponding file paths
categories = {
    'Laptops': 'dataset/laptop_embeddings.npy',
    'smartphone': 'dataset/smartphone_embeddings.npy',
    'Basic Cases': 'dataset/cases_embeddings.npy',
    'Headphone': 'dataset/headphone_embeddings.npy',
    'Laptop Bags': 'dataset/bag_embeddings.npy',
    'Screen Protector': 'dataset/screen_embeddings.npy',
    'Phone Charger': 'dataset/phone_charger_embeddings.npy',
    'mouse': 'dataset/mouse_embeddings.npy',
    'Laptop Charger': 'dataset/laptop_charger_embeddings.npy'
}

# Generate embeddings for each category
for category, path in categories.items():
    generate_embeddings(dataset, category, path)