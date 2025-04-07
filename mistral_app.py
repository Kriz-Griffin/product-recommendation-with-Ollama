import streamlit as st
import pandas as pd
import ollama
import re


def run():
    st.title("Mistral Recommendations")
    st.write("Recommendations")
    # Function to load dataset
    def load_data():
        """Load dataset with error handling."""
        try:
            return pd.read_csv("dataset/boston data 1.csv")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()

    dataset = load_data()

    # Function to load unique categories
    def load_categories():
        """Extract unique categories from dataset."""
        if not dataset.empty and "category" in dataset.columns:
            return sorted(dataset["category"].dropna().unique().tolist())
        return []

    categories = load_categories()
    # User input for category selection
    category = st.selectbox("Select a Category:", categories)

    # User input for search query
    query = st.text_input("Enter your search query:")

    if st.button("Get Recommendations"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            st.info("Fetching recommendations... Please wait.")

            # Function to get product recommendations
            def get_recommendation(query, dataset, category):
                """Fetch product recommendations using Ollama."""
                
                # Filter dataset for the selected category
                dataset = dataset[dataset['category'] == category].copy()

                if dataset.empty:
                    st.error("No products found for this category.")
                    return ""

                # Limit the dataset sample
                data_sample = dataset[['product_id', 'product_title','brand','model', 'features', 'price', 'rating', 'review_count']].to_string(index=False)

                # Create a structured JSON-based prompt
                prompt = f"""
                You are a product recommendation assistant. 
                A user is searching for a product based on the following query: "{query}". 
                Below is a sample of available products:

                {data_sample}

                Based on the query, recommend the top 10 most suitable products **STRICTLY** in JSON format:
                [
                    {{"product_id": "Product ID"}}
                ]
                Only return JSON, nothing else.
                """

                try:
                    # Send request to Ollama
                    response = ollama.chat(
                        model="mistral:7b",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    if not response or "message" not in response or "content" not in response["message"]:
                        st.error("Invalid response from Ollama API.")
                        return ""

                    return response["message"]["content"]

                except Exception as e:
                    st.error(f"Error fetching recommendations: {e}")
                    return ""

            # Fetch recommendations
            response_text = get_recommendation(query, dataset, category)

            # Extract product IDs using regex
            pattern = r'"product_id":\s*"([^"]+)"'
            product_ids = re.findall(pattern, response_text)

            if not product_ids:
                st.warning("No products recommended. Try refining your query.")
            else:
                st.success(f"Extracted {len(product_ids)} recommended product(s).")

                # Filter dataset based on extracted product IDs
                filtered_dataset = dataset[dataset['product_id'].isin(product_ids)]

                # Display results
                st.subheader("📌 Recommended Products")
                # Display product images if available
                st.subheader("📸 Product Images")
                for _, row in filtered_dataset.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 3])  # Adjust column width
                        with col1:
                        # Validate image URL
                            image_url = row.get("image_url", "").strip()
                            if image_url and image_url.lower() != "n/a":
                                st.image(image_url, width=120)  # Display product image
                            else:
                                st.warning("🚫 No Image Available")
                    with col2:
                        st.markdown(f"### {row['product_title']}")
                        st.markdown(f"**Brand:** {row['brand']}")
                        st.markdown(f"**Model:** {row['model']}")
                        st.markdown(f"**Features:** {row['features']}")
                        st.markdown(f"💰 **Price:** ₹{row['price']}")
                        st.markdown(f"⭐ **Rating:** {row['rating']} ({row['review_count']} reviews)")

                st.dataframe(filtered_dataset[['product_id', 'category', 'product_title','features', 'brand', 'price', 'rating', 'review_count']])

if __name__ == "__main__":
    run()
