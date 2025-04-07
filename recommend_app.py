import load_embed as load_embed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st


def run():
    # Load dataset with error handling
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("dataset/boston data 1.csv")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()  # Return empty dataframe if there's an error

    dataset = load_data()

    # Load pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Function to load available categories
    def load_categories(dataset):
        if 'category' in dataset.columns:
            return dataset['category'].unique()
        else:
            st.error("Dataset does not contain the required 'category' column.")
            return []  # Return an empty list if 'category' column is missing

    categories = load_categories(dataset)



    # Streamlit UI Components
    st.title("🔍 Product Recommendation System")

    # Sidebar - Category Selection
    if len(categories) > 0:  # Ensure categories are not empty
        category = st.selectbox("📌 Select a Category", categories)
    else:
        st.warning("No categories found in dataset.")
        return


    # Text Input for Query
    query = st.text_input("📝 Enter your product search query:")

    if st.button("🔍 Find Similar Products") and query:
        if not category:
            st.warning("Please select a specific category.")
            return

        try:
            # Load precomputed embeddings
            embedding = load_embed.load_embedding(category)

            # Filter dataset for selected category
            filtered_dataset = dataset[dataset["category"] == category].reset_index(drop=True)

            # Compute query embedding
            embedding_query = model.encode([query])

            # Compute cosine similarity
            similarities = cosine_similarity(embedding_query, embedding)

            # Sort products by similarity
            sim_scores = list(enumerate(similarities[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_n = 30  # Top 30 recommendations
            sim_scores = sim_scores[:top_n]
            product_indices = [i[0] for i in sim_scores]

            # Retrieve recommended products
            recommended_products = filtered_dataset.loc[product_indices,
                                                         ['product_id', 'product_title', 'category', 'brand', 'rating', 'price', 'review_count','image_url']]
            recommended_products = recommended_products.sort_values(by=['rating', 'review_count'], ascending=[False, False])


            # Display results
            st.subheader("🎯 Top Matching Products:")
            for i, (_, row) in enumerate(recommended_products.iterrows()):
                product_title = row.get("product_title", "Unknown")
                brand = row.get("brand", "Unknown")
                price = row.get("price", "N/A")
                rating = row.get("rating", "N/A")
                similarity_score = sim_scores[i][1]

                st.write(f"**{i+1}. {product_title}**")
                st.write(f"📌 Brand: {brand} | 💰 Price: {price} | ⭐ Rating: {rating} | 🔗 Similarity: {similarity_score:.4f}")

                # Display image if available
                if "image_url" in row and isinstance(row["image_url"], str):
                    st.image(row["image_url"], caption=product_title, width=200)

        except Exception as e:
            st.error(f"Error retrieving product recommendations: {e}")


if __name__ == "__main__":
    run()
