import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import messagebox, font

class DataProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        print("Dataset loaded. First few rows:")
        print(self.data.head())  # Print the first few rows to debug
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
        self.generate_embeddings()

    def generate_embeddings(self):
        # Generate embeddings for the product descriptions
        self.data['DescriptionEmbedding'] = self.data['Description'].apply(
            lambda x: torch.tensor(self.model.encode(x)) if isinstance(x, str) and x.strip() else torch.zeros(self.model.get_sentence_embedding_dimension())
        )
        print("Embeddings generated.")

    def find_similar_product(self, product_name):
        # Normalize the input product name by removing quotation marks and stripping whitespace
        product_name = product_name.replace('"', '').strip().lower()
        print(f"Searching for product: '{product_name}'")  # Debugging print statement

        # Normalize product names in the dataset by removing quotation marks and stripping whitespace
        self.data['NormalizedProductName'] = self.data['ProductName'].str.replace('"', '').str.strip().str.lower()
        product_data = self.data[self.data['NormalizedProductName'] == product_name]

        print("Product data found:")
        print(product_data)  # Debugging print statement

        if product_data.empty:
            print(f"No product found for: '{product_name}'")  # Debugging print statement
            return None  # No product found with the given name

        product_description = product_data.iloc[0]['Description']
        product_embedding = self.model.encode(product_description)

        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(product_embedding,
                                             torch.stack(self.data['DescriptionEmbedding'].tolist()))
        
        # Get the index of the most similar product (excluding itself)
        similar_idx = torch.argmax(similarities[0]).item()

        # Check if the similar product is the same as the input product
        if self.data.iloc[similar_idx]['NormalizedProductName'] == product_data['NormalizedProductName'].values[0]:
            # If the most similar product is itself, find the next most similar one
            similarities[0][similar_idx] = -1  # Exclude itself by setting its similarity to -1
            similar_idx = torch.argmax(similarities[0]).item()
        
        similar_product_name = self.data.iloc[similar_idx]['ProductName']
        
        print(f"Similar product found: {similar_product_name}")  # Debugging print statement
        return similar_product_name


class ProductSimilarityGUI:
    def __init__(self, master, data_processor):
        self.master = master
        self.data_processor = data_processor
        self.master.title("Product Similarity Finder")
        self.master.geometry("400x300")  # Set window size
        self.master.configure(bg="#f0f0f0")  # Set background color

        # Set font styles
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=12)
        self.result_font = font.Font(family="Helvetica", size=12, weight="bold")

        # Create widgets
        self.label = tk.Label(master, text="Enter Product Name:", bg="#f0f0f0", font=self.label_font)
        self.label.pack(pady=(20, 5))  # Add vertical padding

        self.entry = tk.Entry(master, width=40, font=self.label_font)
        self.entry.pack(pady=(0, 20))  # Add vertical padding

        self.find_button = tk.Button(master, text="Find Similar Product", command=self.find_similar,
                                      bg="#007bff", fg="white", font=self.label_font, padx=10, pady=5)
        self.find_button.pack(pady=(0, 10))  # Add vertical padding

        self.result_label = tk.Label(master, text="", bg="#f0f0f0", font=self.result_font)
        self.result_label.pack(pady=(5, 20))  # Add vertical padding

    def find_similar(self):
        product_name = self.entry.get()
        similar_product_name = self.data_processor.find_similar_product(product_name)
        
        if similar_product_name:
            self.result_label.config(text=f"Similar Product: {similar_product_name}")
        else:
            messagebox.showinfo("Product Not Found", f"No similar product found for: '{product_name}'")


if __name__ == "__main__":
    # Load the dataset
    filepath = "NLP_Task_Dataset.csv"  # Update with your dataset path
    data_processor = DataProcessor(filepath)

    # Initialize the GUI
    root = tk.Tk()
    gui = ProductSimilarityGUI(root, data_processor)
    root.mainloop()
