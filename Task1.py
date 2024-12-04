import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox, ttk
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = CountVectorizer()

    def load_data(self):
        data = pd.read_csv(self.filepath)
        return data

    def preprocess_text(self, text):
        text = text.lower()
        return text.strip()

    def train_model(self):
        self.data['Cleaned_Description'] = self.data['Description'].apply(self.preprocess_text)
        self.data.dropna(subset=['Cleaned_Description', 'SubCategory'], inplace=True)

        X = self.vectorizer.fit_transform(self.data['Cleaned_Description']).toarray()
        y = self.data['SubCategory']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # Identify misclassifications using boolean indexing
        misclassifications = pd.DataFrame({
            'Description': self.data.loc[y_test.index, 'Description'],
            'Actual': y_test,
            'Predicted': y_pred
        })

        # Filter misclassifications
        difficult_cases = misclassifications[misclassifications['Actual'] != misclassifications['Predicted']]
        
        # Display difficult cases
        print("Difficult-to-Classify Cases:")
        print(difficult_cases)

    def predict(self, description):
        cleaned_description = self.preprocess_text(description)
        description_vector = self.vectorizer.transform([cleaned_description]).toarray()
        predicted_subcategory = self.model.predict(description_vector)
        return predicted_subcategory[0]

class ProductClassifierApp:
    def __init__(self, master, data_processor):
        self.master = master
        self.data_processor = data_processor

        # Set up the GUI layout
        self.master.title("Product Classifier")
        self.master.geometry("400x300")
        self.master.configure(bg="#f0f0f0")

        # Create a frame for better layout
        self.frame = tk.Frame(self.master, bg="#f0f0f0")
        self.frame.pack(pady=20)

        # Title Label
        self.title_label = tk.Label(self.frame, text="Product Classifier", font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        # Description Input
        self.label = tk.Label(self.frame, text="Enter Product Description:", bg="#f0f0f0")
        self.label.pack(pady=5)

        self.description_entry = tk.Entry(self.frame, width=50, font=("Arial", 12))
        self.description_entry.pack(pady=5)

        # Classify Button
        self.classify_button = tk.Button(self.frame, text="Classify", command=self.classify_description, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.classify_button.pack(pady=10)

        # Result Label
        self.result_label = tk.Label(self.frame, text="", bg="#f0f0f0", font=("Arial", 12, "italic"))
        self.result_label.pack(pady=20)

    def classify_description(self):
        description = self.description_entry.get()
        if description:
            predicted_subcategory = self.data_processor.predict(description)
            self.result_label.config(text=f"Predicted Subcategory: {predicted_subcategory}")
        else:
            messagebox.showwarning("Input Error", "Please enter a product description.")

if __name__ == "__main__":
    filepath = 'NLP_Task_Dataset.csv'  # Update this path
    data_processor = DataProcessor(filepath)

    # Train the model
    data_processor.train_model()

    # Start the GUI application
    root = tk.Tk()
    app = ProductClassifierApp(root, data_processor)
    root.mainloop()
