import os
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from sentence_transformers import SentenceTransformer
import faiss
# import numpy as np
import torch
import time
import threading
import pandas as pd
import ollama
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models_config import CHAT_MODELS, DEFAULT_MODEL, EmbeddingModelList


"""
sentence_transformers / faiss  / torch


sentence_transformers, faiss, torch
custom / HuggingFace embeddings,
large-scale vector search,
training/fine-tuning or direct ML work.
faiss works very will on GPU Ubuntu/Linux.

Train your own embedding model
Fine-tune models on your domain
Run large transformer models directly with transformers + torch
Need GPU-accelerated FAISS indexes

Use FAISS when:
You have lots of chunks (tens of thousands → millions+)
You need fast search & low memory
You’re on Linux (FAISS support is best there)
You want approximate nearest neighbor search, complex indexes, etc.

torch (PyTorch)
Core deep-learning framework.
Used by:
sentence_transformers
Many HuggingFace models
Needed when you:
Train your own models
Run transformers directly (not via Ollama)
Fine-tune embeddings / LLMs
In your current flow:
Ollama already runs the heavy model stuff (C++/CUDA, etc.)
You’re not training or fine-tuning locally
➡️ No need for PyTorch right now.
"""


MainModel = DEFAULT_MODEL #"deepseek-r1:1.5b"
EmbeddingModel = 'all-MiniLM-L6-v2'
# MainModelList = ["phi3:mini","deepseek-r1:1.5b", "deepseek-r1:7b", "gemma3", "gemma3:12b", "gemma-asil", "mistral", "qwen3:1.7b"]
MainModelList = CHAT_MODELS
# EmbeddingModelList = ["all-minilm:l12-v2", "mxbai-embed-large", "nomic-embed-text"]
# EmbeddingModelList =EmbeddingModelList
FormTitle = "RAG Pipeline GUI : Ollama (CSV Files) : sentence_transformers / faiss  / torch"

RAG_DATA_0_FOLDER_NAME = "rag_data"
RAG_DATA_1_FOLDER_NAME = "my_csvs"
RAG_DATA_2_FOLDER_NAME = "docs"

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.set_title(FormTitle)
        self.root.geometry("1000x700")

        # Define paths
        '''
        self.base_dir = os.path.expanduser("./rag_data") #("~/Desktop/Tests/LLM_TESTS/deepseek_rag_gpu")
        # self.base_dir = os.path.abspath("rag_data")
        self.csv_dir = os.path.join(self.base_dir, "my_csvs/docs")
        self.files_dir = os.path.join(self.base_dir, "my_csvs")
        self.csv_chunks_txt = os.path.join(self.files_dir, "csv_chunks.txt")
        # self.chunks_txt = os.path.join(self.files_dir, "chunks_csvs.txt")
        self.csv_index_faiss = os.path.join(self.files_dir, "csv_index.faiss")
        '''
        self.script_dir = Path(__file__).resolve().parent.parent
        # self.base_dir = os.path.expanduser("./rag_data") #("~/Desktop/Tests/LLM_TESTS/deepseek_rag_gpu")
        self.base_dir = self.script_dir / RAG_DATA_0_FOLDER_NAME
        # self.base_dir = os.path.abspath("rag_data")
        # self.csv_dir = os.path.join(self.base_dir, "my_csvs/docs")
        self.csv_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME
        # self.files_dir = os.path.join(self.base_dir, "my_csvs")
        self.files_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME
        self.csv_chunks_txt = os.path.join(self.files_dir, "csv_chunks.txt")
        # self.chunks_txt = os.path.join(self.files_dir, "chunks_csvs.txt")
        self.csv_index_faiss = os.path.join(self.files_dir, "csv_index.faiss")

        # Ensure my_csvs directory exists
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

        # Default models
        self.embedding_model = EmbeddingModel
        self.main_model = MainModel
        self.available_embedding_models = EmbeddingModelList
        self.available_main_models = MainModelList

        # Load embedder with GPU support
        self.load_embedder()

        # UI Elements
        self.create_widgets()

    def load_embedder(self):
        self.embedder = SentenceTransformer(self.embedding_model)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to('cuda')

    def create_widgets(self):
        # Configure grid for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Buttons Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=0, column=0, pady=10, sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        tk.Button(btn_frame, text="0 Create Sample CSV Files", command=self.create_sample_csvs_threaded).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="1 Extract CSV Data", command=self.extract_csv_folder_threaded).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="2 Build Index", command=self.build_index_threaded).grid(row=0, column=2, padx=5)

        # Main Frame (Query and Output)
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Model Selection Frame
        model_frame = tk.Frame(main_frame)
        model_frame.grid(row=0, column=0, pady=5, sticky="ew")

        tk.Label(model_frame, text="Embedding Model:").pack(side=tk.LEFT, padx=5)
        self.embedding_combo = ttk.Combobox(model_frame, values=self.available_embedding_models, state="readonly")
        self.embedding_combo.set(self.embedding_model)
        self.embedding_combo.pack(side=tk.LEFT, padx=5)
        self.embedding_combo.bind("<<ComboboxSelected>>", self.update_embedding_model)

        tk.Label(model_frame, text="Main Model:").pack(side=tk.LEFT, padx=5)
        self.main_combo = ttk.Combobox(model_frame, values=self.available_main_models, state="readonly")
        self.main_combo.set(self.main_model)
        self.main_combo.pack(side=tk.LEFT, padx=5)
        self.main_combo.bind("<<ComboboxSelected>>", self.update_main_model)

        # Query Frame
        query_frame = tk.Frame(main_frame)
        query_frame.grid(row=1, column=0, pady=5, sticky="nsew")
        query_frame.columnconfigure(1, weight=1)
        query_frame.rowconfigure(0, weight=1)

        tk.Label(query_frame, text="Enter Query:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.query_text = tk.Text(query_frame, width=50, height=5, font=("Arial", 12))
        self.query_text.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(query_frame, text="Run Query", command=self.run_query_threaded).grid(row=0, column=2, padx=5, pady=5)

        # Bind Ctrl+Enter to run_query_threaded
        self.query_text.bind("<Control-Return>", lambda event: self.run_query_threaded())
        
        # Output Area
        self.output_text = scrolledtext.ScrolledText(main_frame, width=90, height=20, font=("Arial", 12))
        self.output_text.grid(row=2, column=0, pady=10, sticky="nsew")

    def log(self, message):
        if "Query:" in message:
            self.output_text.insert(tk.END, "\n---------------------------\n")
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def set_title(self, message):
        self.root.title(message)

    def update_embedding_model(self, event):
        new_model = self.embedding_combo.get()
        if new_model != self.embedding_model:
            self.embedding_model = new_model
            self.load_embedder()
            self.log(f"Embedding model updated to: {self.embedding_model}")

    def update_main_model(self, event):
        self.main_model = self.main_combo.get()
        self.log(f"Main model updated to: {self.main_model}")

    def create_sample_csvs_threaded(self):
        threading.Thread(target=self.create_sample_csvs, daemon=True).start()

    def create_sample_csvs(self):
        self.set_title(FormTitle + " : create_sample_csvs ...")
        # Sample CSV 1: Customer Data
        customer_data = [
            {"name": "Alice Johnson", "orders": "Laptop, Mouse", "description": "Customer interested in tech gadgets"},
            {"name": "Bob Smith", "orders": "Keyboard, Monitor", "description": "Customer upgrading home office"},
            {"name": "Charlie Davis", "orders": "Smartphone, Tablet", "description": "Customer seeking mobile devices"},
            {"name": "Diana Garcia", "orders": "Headphones, Speakers", "description": "Customer focused on audio equipment"},
        ]
        df1 = pd.DataFrame(customer_data)
        output_path1 = os.path.join(self.csv_dir, "customer_data.csv")
        df1.to_csv(output_path1, index=False)

        # Sample CSV 2: Medical Rules
        medical_data = [
            {"ActivityCode": "00100", "AcceptedDiagnosis": "C07, C08.0, C08.1, C08.9, C79.89, C79.9", "Rule": "MedicalNecessity"},
            {"ActivityCode": "00102", "AcceptedDiagnosis": "Q36.0, Q36.1, Q36.9, Q37.0, Q37.1, Q37.2", "Rule": "MedicalNecessity"},
            {"ActivityCode": "00103", "AcceptedDiagnosis": "H02.121, H02.122, H02.123, H02.124", "Rule": "VPSActivityFound"},
            {"ActivityCode": "00104", "AcceptedDiagnosis": "H02.131, H02.132, H02.133, H02.134", "Rule": "VPSActivityFound"},
        ]
        df2 = pd.DataFrame(medical_data)
        output_path2 = os.path.join(self.csv_dir, "medical_rules.csv")
        df2.to_csv(output_path2, index=False)

        self.log(f"Sample CSV files created in {self.csv_dir}")
        self.set_title(FormTitle)

    def extract_csv_folder_threaded(self):
        threading.Thread(target=self.extract_csv_folder, daemon=True).start()

    def extract_csv_folder(self):
        self.set_title(FormTitle + " : extract_csv_folder ...")
        all_chunks = []
        for csv_file in os.listdir(self.csv_dir):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(self.csv_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    # Combine all columns into a single string per row
                    for _, row in df.iterrows():
                        # Join all non-null column values into a single string
                        chunk = " | ".join(str(val) for val in row if pd.notna(val))
                        if chunk.strip():
                            all_chunks.append(chunk)
                except Exception as e:
                    self.log(f"Error processing {csv_file}: {e}")

        with open(self.csv_chunks_txt, "w") as f:
            for chunk in all_chunks:
                f.write(chunk.strip() + "\n\n")

        self.log(f"Extracted {len(all_chunks)} chunks from CSV files in {self.csv_dir}")
        self.set_title(FormTitle)

    def build_index_threaded(self):
        threading.Thread(target=self.build_index, daemon=True).start()

    def build_index(self):
        self.set_title(FormTitle + " : build_index ...")
        with open(self.csv_chunks_txt, "r") as f:
            chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, batch_size=32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, self.csv_index_faiss)
        # with open(self.chunks_txt, "w") as f:
            # f.write("\n\n".join(chunks))

        self.log("FAISS index created at: " + self.csv_index_faiss)
        self.set_title(FormTitle)

    def run_query_threaded(self):
        threading.Thread(target=self.run_query, daemon=True).start()

    def run_query(self):
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        self.set_title(FormTitle + " : run_query ...")

        # Load index and chunks if not already loaded
        if not hasattr(self, 'index'):
            self.index = faiss.read_index(self.csv_index_faiss)
            # with open(self.chunks_txt, "r") as f:
            with open(self.csv_chunks_txt, "r") as f:
                self.chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

        # Retrieve and generate response
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k=1)
        context = self.chunks[indices[0][0]]
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        start = time.time()
        try:
            response = ollama.chat(
                model=self.main_model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]
            latency = time.time() - start
            self.log(f"Query: {query}")
            self.log(f"Answer: {answer}")
            self.log(f"Latency: {latency:.3f} seconds")
            self.log(f"MainModel / EmbeddingModel: {self.main_model} / {self.embedding_model}\n")
            self.set_title(FormTitle)
        except Exception as e:
            self.log(f"Error querying Ollama: {e}")
            self.set_title(FormTitle)

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()