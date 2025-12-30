import os
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from anyio import Path
import numpy as np
import time
import threading
import pandas as pd
import ollama
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models_config import CHAT_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL, EMBEDDING_MODELS

"""
Ollama embeddings + NumPy is simpler, portable, Windows-friendly
"""

# Tkinter + CSV → chunks → Ollama Embeddings + NumPy similarity → Ollama Chat.

# ---------------- CONFIG ----------------
# 

MainModel = DEFAULT_MODEL #"deepseek-r1:1.5b"

# An Ollama **embedding** model (must be pulled via `ollama pull`).
# Good options: "all-minilm", "mxbai-embed-large", "nomic-embed-text"
# EMBEDDING_MODEL = "all-minilm"

# MainModelList = [
#     "deepseek-r1:1.5b",
#     "phi3:mini",
#     "deepseek-r1:7b",
#     "gemma3",
#     "gemma3:12b",
#     "mistral",
#     "qwen3:1.7b",
# ]

MainModelList = CHAT_MODELS
# EMBEDDING_MODELS = [
#     "all-minilm",
#     "mxbai-embed-large",
#     "nomic-embed-text",
# ]

FormTitle = "RAG Pipeline GUI : Ollama (CSV Files) : Ollama embeddings + NumPy "

RAG_DATA_0_FOLDER_NAME = "rag_data"
RAG_DATA_1_FOLDER_NAME = "my_csvs"
RAG_DATA_2_FOLDER_NAME = "docs"

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.set_title(FormTitle)
        self.root.geometry("1000x700")

        # ---------- Paths ----------
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_dir, "rag_data")

        self.csv_dir = os.path.join(self.base_dir, "my_csvs", "docs")
        self.files_dir = os.path.join(self.base_dir, "my_csvs")
        self.csv_chunks_txt = os.path.join(self.files_dir, "csv_chunks.txt")
        # self.chunks_txt = os.path.join(self.files_dir, "chunks_csvs.txt")
        # store embeddings in simple NumPy file instead of FAISS index
        self.csv_index_npy = os.path.join(self.files_dir, "csv_index.npy")
        '''

        # script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_dir = Path(__file__).resolve().parent.parent
        # self.base_dir = os.path.join(script_dir, "rag_data")
        self.base_dir = self.script_dir / RAG_DATA_0_FOLDER_NAME

        # self.csv_dir = os.path.join(self.base_dir, "my_csvs", "docs")
        self.csv_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME

        # self.files_dir = os.path.join(self.base_dir, "my_csvs")
        self.files_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME
        self.csv_chunks_txt = os.path.join(self.files_dir, "csv_chunks.txt")
        # self.chunks_txt = os.path.join(self.files_dir, "chunks_csvs.txt")
        # store embeddings in simple NumPy file instead of FAISS index
        self.csv_index_npy = os.path.join(self.files_dir, "csv_index.npy")

        # Ensure directories exist
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Default models
        self.embedding_model = EMBEDDING_MODEL
        self.main_model = MainModel
        self.available_embedding_models = EMBEDDING_MODELS
        self.available_main_models = MainModelList

        # Embedding matrix + chunks (lazy-loaded)
        self.embeddings = None  # shape: (num_chunks, dim)
        self.chunks = None

        # UI Elements
        self.create_widgets()

    # ---------- UI ----------

    def create_widgets(self):
        # Configure grid for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Buttons Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=0, column=0, pady=10, sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        tk.Button(
            btn_frame,
            text="0 Create Sample CSV Files",
            command=self.create_sample_csvs_threaded,
        ).grid(row=0, column=0, padx=5)
        tk.Button(
            btn_frame,
            text="1 Extract CSV Data",
            command=self.extract_csv_folder_threaded,
        ).grid(row=0, column=1, padx=5)
        tk.Button(
            btn_frame,
            text="2 Build Index",
            command=self.build_index_threaded,
        ).grid(row=0, column=2, padx=5)

        # Main Frame (Query and Output)
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Model Selection Frame
        model_frame = tk.Frame(main_frame)
        model_frame.grid(row=0, column=0, pady=5, sticky="ew")

        tk.Label(model_frame, text="Embedding Model:").pack(side=tk.LEFT, padx=5)
        self.embedding_combo = ttk.Combobox(
            model_frame,
            values=self.available_embedding_models,
            state="readonly",
            width=22,
        )
        self.embedding_combo.set(self.embedding_model)
        self.embedding_combo.pack(side=tk.LEFT, padx=5)
        self.embedding_combo.bind("<<ComboboxSelected>>", self.update_embedding_model)

        tk.Label(model_frame, text="Main Model:").pack(side=tk.LEFT, padx=5)
        self.main_combo = ttk.Combobox(
            model_frame,
            values=self.available_main_models,
            state="readonly",
            width=22,
        )
        self.main_combo.set(self.main_model)
        self.main_combo.pack(side=tk.LEFT, padx=5)
        self.main_combo.bind("<<ComboboxSelected>>", self.update_main_model)

        # Query Frame
        query_frame = tk.Frame(main_frame)
        query_frame.grid(row=1, column=0, pady=5, sticky="nsew")
        query_frame.columnconfigure(1, weight=1)
        query_frame.rowconfigure(0, weight=1)

        tk.Label(
            query_frame, text="Enter Query:", font=("Arial", 12)
        ).grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.query_text = tk.Text(query_frame, width=50, height=5, font=("Arial", 12))
        self.query_text.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(
            query_frame,
            text="Run Query",
            command=self.run_query_threaded,
        ).grid(row=0, column=2, padx=5, pady=5)

        # Ctrl+Enter to run query
        self.query_text.bind(
            "<Control-Return>", lambda event: self.run_query_threaded()
        )

        # Output Area
        self.output_text = scrolledtext.ScrolledText(
            main_frame, width=90, height=20, font=("Arial", 12)
        )
        self.output_text.grid(row=2, column=0, pady=10, sticky="nsew")

    # ---------- Helpers ----------

    def log(self, message):
        if "Query:" in message:
            self.output_text.insert(tk.END, "\n---------------------------\n")
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def set_title(self, message):
        self.root.title(message)

    def update_embedding_model(self, event):
        self.embedding_model = self.embedding_combo.get()
        self.embeddings = None  # force rebuild if changed
        self.log(f"Embedding model updated to: {self.embedding_model}")

    def update_main_model(self, event):
        self.main_model = self.main_combo.get()
        self.log(f"Main model updated to: {self.main_model}")

    # ---------- Buttons: threaded wrappers ----------

    def create_sample_csvs_threaded(self):
        threading.Thread(target=self.create_sample_csvs, daemon=True).start()

    def extract_csv_folder_threaded(self):
        threading.Thread(target=self.extract_csv_folder, daemon=True).start()

    def build_index_threaded(self):
        threading.Thread(target=self.build_index, daemon=True).start()

    def run_query_threaded(self):
        threading.Thread(target=self.run_query, daemon=True).start()

    # ---------- Step 0: create sample CSVs ----------

    def create_sample_csvs(self):
        self.set_title(FormTitle + " : create_sample_csvs ...")

        customer_data = [
            {
                "name": "Alice Johnson",
                "orders": "Laptop, Mouse",
                "description": "Customer interested in tech gadgets",
            },
            {
                "name": "Bob Smith",
                "orders": "Keyboard, Monitor",
                "description": "Customer upgrading home office",
            },
            {
                "name": "Charlie Davis",
                "orders": "Smartphone, Tablet",
                "description": "Customer seeking mobile devices",
            },
            {
                "name": "Diana Garcia",
                "orders": "Headphones, Speakers",
                "description": "Customer focused on audio equipment",
            },
        ]
        df1 = pd.DataFrame(customer_data)
        output_path1 = os.path.join(self.csv_dir, "customer_data.csv")
        df1.to_csv(output_path1, index=False)

        medical_data = [
            {
                "ActivityCode": "00100",
                "AcceptedDiagnosis": "C07, C08.0, C08.1, C08.9, C79.89, C79.9",
                "Rule": "MedicalNecessity",
            },
            {
                "ActivityCode": "00102",
                "AcceptedDiagnosis": "Q36.0, Q36.1, Q36.9, Q37.0, Q37.1, Q37.2",
                "Rule": "MedicalNecessity",
            },
            {
                "ActivityCode": "00103",
                "AcceptedDiagnosis": "H02.121, H02.122, H02.123, H02.124",
                "Rule": "VPSActivityFound",
            },
            {
                "ActivityCode": "00104",
                "AcceptedDiagnosis": "H02.131, H02.132, H02.133, H02.134",
                "Rule": "VPSActivityFound",
            },
        ]
        df2 = pd.DataFrame(medical_data)
        output_path2 = os.path.join(self.csv_dir, "medical_rules.csv")
        df2.to_csv(output_path2, index=False)

        self.log(f"Sample CSV files created in {self.csv_dir}")
        self.set_title(FormTitle)

    # ---------- Step 1: extract CSV rows into text chunks ----------

    def extract_csv_folder(self):
        self.set_title(FormTitle + " : extract_csv_folder ...")
        all_chunks = []

        for csv_file in os.listdir(self.csv_dir):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(self.csv_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        chunk = " | ".join(
                            str(val) for val in row if pd.notna(val)
                        )
                        if chunk.strip():
                            all_chunks.append(chunk)
                except Exception as e:
                    self.log(f"Error processing {csv_file}: {e}")

        with open(self.csv_chunks_txt, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk.strip() + "\n\n")

        self.log(
            f"Extracted {len(all_chunks)} chunks from CSV files in {self.csv_dir}"
        )
        self.set_title(FormTitle)

    # ---------- Step 2: build embedding index using Ollama ----------

    def build_index(self):
        self.set_title(FormTitle + " : build_index ...")

        if not os.path.exists(self.csv_chunks_txt):
            self.log("No csv_chunks.txt found. Run 'Extract CSV Data' first.")
            self.set_title(FormTitle)
            return

        with open(self.csv_chunks_txt, "r", encoding="utf-8") as f:
            chunks = [
                chunk.strip()
                for chunk in f.read().split("\n\n")
                if chunk.strip()
            ]

        if not chunks:
            self.log("No chunks found to index.")
            self.set_title(FormTitle)
            return

        embeddings = []
        self.log(
            f"Building embeddings with model: {self.embedding_model} "
            f"for {len(chunks)} chunks..."
        )

        for i, chunk in enumerate(chunks, start=1):
            try:
                emb_resp = ollama.embeddings(
                    model=self.embedding_model, prompt=chunk
                )
                emb = emb_resp["embedding"]
                embeddings.append(emb)
            except Exception as e:
                self.log(f"Error embedding chunk {i}: {e}")

        if not embeddings:
            self.log("No embeddings were created. Check embedding model.")
            self.set_title(FormTitle)
            return

        embeddings = np.array(embeddings, dtype="float32")
        np.save(self.csv_index_npy, embeddings)

        # with open(self.chunks_txt, "w", encoding="utf-8") as f:
        #     f.write("\n\n".join(chunks))

        self.embeddings = embeddings
        self.chunks = chunks

        self.log("Index built and saved (NumPy) at: " + self.csv_index_npy)
        self.set_title(FormTitle)

    # ---------- Step 3: run query ----------

    def load_index_if_needed(self):
        if self.embeddings is not None and self.chunks is not None:
            return True

        if not os.path.exists(self.csv_index_npy) or not os.path.exists(
            self.csv_chunks_txt #self.chunks_txt
        ):
            self.log(
                "Index or chunks file missing. Please run 'Build Index' first."
            )
            return False

        self.embeddings = np.load(self.csv_index_npy)
        # with open(self.chunks_txt, "r", encoding="utf-8") as f:
        with open(self.csv_chunks_txt, "r", encoding="utf-8") as f:
            self.chunks = [
                chunk.strip()
                for chunk in f.read().split("\n\n")
                if chunk.strip()
            ]
        return True

    def run_query(self):
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return

        # Load index and chunks if not already loaded
        if not self.load_index_if_needed():
            return

        self.set_title(FormTitle + " : run_query ...")

        # 1) Embed the query with Ollama
        try:
            q_emb_resp = ollama.embeddings(
                model=self.embedding_model, prompt=query
            )
            q_emb = np.array(q_emb_resp["embedding"], dtype="float32")
        except Exception as e:
            self.log(f"Error embedding query: {e}")
            self.set_title(FormTitle)
            return

        # 2) Cosine similarity search over embeddings matrix
        emb = self.embeddings  # (N, D)
        # normalize
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        q_norm = q_emb / np.linalg.norm(q_emb)
        sims = emb_norm @ q_norm  # (N,)

        # get top-1 (you can change to top-k)
        # top_idx = int(np.argmax(sims))
        # context = self.chunks[top_idx]

        # get top-k = 3
        top_k = 3
        top_idx = np.argsort(sims)[-top_k:][::-1]
        context = "\n\n".join(self.chunks[i] for i in top_idx)
        
        # For debugging, show similarity scores
        # top_contexts = [self.chunks[i] for i in top_idx]
        # context = "\n\n---\n\n".join(
        #     f"[Chunk {rank+1} | sim={sims[i]:.4f}]\n{self.chunks[i]}"
        #     for rank, i in enumerate(top_idx)
        # )

        # 3) Formulate prompt and query main model
        # prompt = (
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     f"Answer clearly and concisely based only on the context above."
        # )

        # prompt = (
        #     "Answer ONLY using the Context.\n"
        #     "If the Context does not contain the answer, reply exactly:\n"
        #     "I don't know based on the provided context.\n\n"
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     "Answer:"
        # )

        # here is ageneral question (6+9)
        # or the RAG depend on the context (RAG with CSV data)
        # prompt = (
        #     "You are a hybrid RAG assistant.\n\n"
        #     "Decision rules:\n"
        #     "1) If the Context contains information that directly answers the Question, answer using ONLY the Context.\n"
        #     "2) If the Question is a general knowledge question and the Context is irrelevant or unrelated, ignore the Context and answer normally.\n"
        #     "3) If the Context is related but does NOT contain the answer, reply exactly:\n"
        #     "I don't know based on the provided context.\n\n"
        #     "IMPORTANT:\n"
        #     "- Do NOT mix Context knowledge with general knowledge.\n"
        #     "- Use Context ONLY in case (1).\n\n"
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     "Answer:"
        # )

        prompt = (
            "You are a hybrid RAG assistant.\n\n"
            "Decision rules:\n"
            "1) If the Context contains information that directly answers the Question, answer using ONLY the Context.\n"
            "2) If the Question is a general knowledge question and the Context is irrelevant or unrelated, ignore the Context and answer normally.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )        

        start = time.time()
        try:
            response = ollama.chat(
                model=self.main_model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response["message"]["content"]
            latency = time.time() - start

            self.log(f"Query: {query}")
            self.log(f"Context (top-3 chunk): {context}")
            self.log(f"Answer: {answer}")
            self.log(f"Latency: {latency:.3f} seconds")
            self.log(
                f"MainModel / EmbeddingModel: {self.main_model} / {self.embedding_model}\n"
            )
            self.set_title(FormTitle)
        except Exception as e:
            self.log(f"Error querying Ollama: {e}")
            self.set_title(FormTitle)


if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
