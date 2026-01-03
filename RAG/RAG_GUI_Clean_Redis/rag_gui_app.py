from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from rag_engine_qdrant import RAGEngineQdrant
from redis_store import RedisStore, RedisStoreConfig, format_history_for_prompt, default_user_id

from rag_settings import (
    ALL_PROVIDERS,
    APP_TITLE,
    DEFAULT_ANSWER_PROMPT_TEMPLATE,
    DEFAULT_APP_ID,
    DEFAULT_COLLECTION_PREFIX,
    DEFAULT_DOCS_DIR,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_ENV_PATH,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_PROVIDER,
    DEFAULT_QDRANT_API_KEY,
    DEFAULT_QDRANT_URL,
    DEFAULT_REDIS_URL,
    DEFAULT_RERANK_PROMPT_TEMPLATE,
    DEFAULT_SESSION_ID,
    DEFAULT_TEXT_GROUP_LINES,
    DEFAULT_TOPK_RETRIEVE,
    DEFAULT_TOPK_USE,
    DEFAULT_USER_ID,
    GEMINI_PROVIDER,
    OLLAMA_PROVIDER,
    OPENAI_PROVIDER,
    RAG_DATA_0_FOLDER_NAME,
    RAG_DATA_1_FOLDER_NAME,
    RAG_DATA_2_FOLDER_NAME,
    SETTINGS_FILE_NAME,
    detect_provider_for_model,
    load_env,
    load_settings,
    default_settings,
    resolve_docs_dir,
    safe_bool,
    safe_int,
    save_settings,
    CHAT_MODELS,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    EMBEDDING_MODELS_GEMINI,
    EMBEDDING_MODELS_OPENAI,
    CHAT_MODELS_GEMINI,
    CHAT_MODELS_OPENAI,
)


class RAGAppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x760")

        # confirm export on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Paths
        self.settings_path = Path(__file__).resolve().parent / SETTINGS_FILE_NAME
        self.script_dir = Path(__file__).resolve().parent.parent
        self.base_dir = self.script_dir / RAG_DATA_0_FOLDER_NAME
        self.csv_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data folder: {self.csv_dir}")

        # --------- LOAD SETTINGS (or defaults) BEFORE creating tk variables ----------
        loaded = self._load_settings_file()

        # docs folder path from settings
        self.docs_dir_var = tk.StringVar(value=loaded.get("docs_dir", DEFAULT_DOCS_DIR))

        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data folder: {self.csv_dir}")

        # prompt templates from settings
        self.rerank_prompt_var = tk.StringVar(value=loaded.get("rerank_prompt_template", DEFAULT_RERANK_PROMPT_TEMPLATE))
        self.answer_prompt_var = tk.StringVar(value=loaded.get("answer_prompt_template", DEFAULT_ANSWER_PROMPT_TEMPLATE))
        
        self.main_provider_var = tk.StringVar(value=loaded.get("main_provider", DEFAULT_PROVIDER))
        self.embedding_provider_var = tk.StringVar(value=loaded.get("embedding_provider", DEFAULT_PROVIDER))
        self.env_path_var = tk.StringVar(value=loaded.get("env_path", DEFAULT_ENV_PATH))

        # Redis settings
        self.redis_url_var = tk.StringVar(value=loaded.get("redis_url", DEFAULT_REDIS_URL))
        self.app_id_var = tk.StringVar(value=loaded.get("app_id", DEFAULT_APP_ID))
        self.session_id_var = tk.StringVar(value=loaded.get("session_id", DEFAULT_SESSION_ID))
        self.user_id_var = tk.StringVar(value=loaded.get("user_id", DEFAULT_USER_ID))
        self.history_turns_var = tk.IntVar(value=safe_int(loaded.get("history_turns", DEFAULT_HISTORY_TURNS), DEFAULT_HISTORY_TURNS))

        # self._load_env_from_settings()

        # # Models state
        # self.embedding_model = EMBEDDING_MODEL
        # self.main_model = DEFAULT_MODEL

        # Models state (load from file if exists)
        self.embedding_model = loaded.get("embedding_model", EMBEDDING_MODEL)  
        self.main_model = loaded.get("main_model", DEFAULT_MODEL)            

        # # Settings state
        # self.qdrant_url = tk.StringVar(value=DEFAULT_QDRANT_URL)
        # self.collection_prefix = tk.StringVar(value=DEFAULT_COLLECTION_PREFIX)
        # self.topk_retrieve = tk.IntVar(value=DEFAULT_TOPK_RETRIEVE)
        # self.topk_use = tk.IntVar(value=DEFAULT_TOPK_USE)
        # self.enable_rerank = tk.BooleanVar(value=DEFAULT_ENABLE_RERANK)
        # self.text_group_lines = tk.IntVar(value=DEFAULT_TEXT_GROUP_LINES)

        # Settings state (load from file if exists)
        self.qdrant_url = tk.StringVar(value=loaded.get("qdrant_url", DEFAULT_QDRANT_URL))
        self.collection_prefix = tk.StringVar(value=loaded.get("collection_prefix", DEFAULT_COLLECTION_PREFIX))
        self.topk_retrieve = tk.IntVar(value=safe_int(loaded.get("topk_retrieve", DEFAULT_TOPK_RETRIEVE), DEFAULT_TOPK_RETRIEVE))
        self.topk_use = tk.IntVar(value=safe_int(loaded.get("topk_use", DEFAULT_TOPK_USE), DEFAULT_TOPK_USE))
        self.enable_rerank = tk.BooleanVar(value=safe_bool(loaded.get("enable_rerank", DEFAULT_ENABLE_RERANK), DEFAULT_ENABLE_RERANK))
        self.text_group_lines = tk.IntVar(value=safe_int(loaded.get("text_group_lines", DEFAULT_TEXT_GROUP_LINES), DEFAULT_TEXT_GROUP_LINES))

        # self.history_turns = tk.IntVar(value=safe_int(loaded.get("history_turns", DEFAULT_HISTORY_TURNS), DEFAULT_HISTORY_TURNS))

        # # Engine (created lazily/refreshable)
        # self.engine = self._make_engine()

        # Thread-safe UI logging via queue
        self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        # Build UI
        self._build_ui() # here init logging

        self._load_env_from_settings() # load env after UI is built, for logging

        self._init_redis_store()

        # Engine (created lazily/refreshable)
        self.engine = self._make_engine() # here after env load        

        # Ensure combos reflect loaded models (and clamp if invalid)
        if self.embedding_model not in EMBEDDING_MODELS:
            self.embedding_model = EMBEDDING_MODEL
        if self.main_model not in CHAT_MODELS:
            self.main_model = DEFAULT_MODEL
        self.embedding_combo.set(self.embedding_model)
        self.main_combo.set(self.main_model)
        self.engine.set_models(self.embedding_model, self.main_model)

        # Start UI queue pump
        self._pump_ui_queue()

    def _load_env_from_settings(self):
        env_path = load_env(self.env_path_var.get(), script_dir=self.script_dir)
        if env_path is None:
            return
        if env_path.exists():
            self._append_log(f"[ENV] Loaded: {env_path}")
        else:
            self._append_log(f"[ENV] Not found: {env_path}")

    def _init_redis_store(self):
        # keep last init error for UI
        self.redis_init_error = None

        try:
            cfg = RedisStoreConfig(
                redis_url=self.redis_url_var.get(),
                app_id=self.app_id_var.get(),
                # user_id=default_user_id(),
                user_id=self.user_id_var.get(),
                session_id=self.session_id_var.get(),
                history_max_turns=50,
            )

            self.redis_store = RedisStore(cfg)

            # RedisStore may be disabled if redis-py missing or url empty
            if not self.redis_store.enabled:
                self.redis_init_error = self.redis_store.last_error or "Redis disabled"
                self._append_log(f"[Redis] Disabled: {self.redis_init_error}")
                return

            # ping may fail if server not running yet
            if self.redis_store.ping():
                self._append_log(f"[Redis] Connected OK: {self.redis_url_var.get()}")
            else:
                self.redis_init_error = self.redis_store.last_error or "Ping failed"
                self._append_log(f"[Redis] Unreachable: {self.redis_init_error}")

        except Exception as e:
            self.redis_store = None
            self.redis_init_error = str(e)
            self._append_log(f"[Redis] Init error: {self.redis_init_error}")


    def _make_engine(self) -> RAGEngineQdrant:
        eng = RAGEngineQdrant(
            qdrant_url=self.qdrant_url.get(),
            api_key=DEFAULT_QDRANT_API_KEY,
            collection_prefix=self.collection_prefix.get(),
        )
        eng.set_models(self.embedding_model, self.main_model)
        eng.set_prompts(self.rerank_prompt_var.get(), self.answer_prompt_var.get())      
        eng.set_provider(
            main_provider=self.main_provider_var.get(),
            embedding_provider=self.embedding_provider_var.get(),
            # gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        )
        
        return eng

    def _open_docs_dir(self):
        path = self._resolve_docs_dir(self.docs_dir_var.get())
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not path.exists():
            messagebox.showwarning("Warning", f"Folder does not exist:\n{path}")
            return

        try:
            # Windows
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
                return

            # macOS
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
                return

            # Linux
            subprocess.run(["xdg-open", str(path)], check=False)

        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to open folder:\n{path}\n\n{e}")

    def _browse_docs_dir(self):
        # start from current value if possible
        initial = str(self._resolve_docs_dir(self.docs_dir_var.get()))
        if not Path(initial).exists():
            initial = str(self.script_dir)

        selected = filedialog.askdirectory(
            title="Select Docs Folder (CSV/TXT for indexing)",
            initialdir=initial,
            mustexist=True,
        )
        if not selected:
            return

        # Prefer saving relative paths (cleaner + portable)
        try:
            rel = Path(selected).resolve().relative_to(self.script_dir.resolve())
            self.docs_dir_var.set(str(rel))
        except Exception:
            # fallback: store absolute
            self.docs_dir_var.set(str(Path(selected).resolve()))

        # Update live folder target
        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self._append_log(f"[Settings] Docs folder set to: {self.csv_dir}")

    # def _resolve_docs_dir(self, p: str) -> Path:
    #     p = (p or "").strip()
    #     if not p:
    #         p = DEFAULT_DOCS_DIR
    #     path = Path(p)
    #     if not path.is_absolute():
    #         path = self.script_dir / path
    #     return path

    def _resolve_docs_dir(self, p: str) -> Path:
        p = (p or "").strip()

        if not p:
            return Path(DEFAULT_DOCS_DIR)

        path = Path(p)

        # If user provided absolute path → use it as-is
        if path.is_absolute(): # D:\LLM\LLMs_Tests\RAG\rag_data\my_csvs\docs
            return path

        # Otherwise treat it as relative to project root #rag_data\my_csvs\docs
        return (self.script_dir / path).resolve()

    # ---------- UI Layout ----------
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Label(toolbar, text="Embedding Model:").pack(side=tk.LEFT, padx=(0, 6))
        self.embedding_combo = ttk.Combobox(
            toolbar, values=EMBEDDING_MODELS, state="readonly", width=28
        )
        self.embedding_combo.set(self.embedding_model)
        self.embedding_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.embedding_combo.bind("<<ComboboxSelected>>", self._on_embedding_changed)

        tk.Label(toolbar, text="Main Model:").pack(side=tk.LEFT, padx=(0, 6))
        self.main_combo = ttk.Combobox(
            toolbar, values=CHAT_MODELS, state="readonly", width=28
        )
        self.main_combo.set(self.main_model)
        self.main_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.main_combo.bind("<<ComboboxSelected>>", self._on_main_changed)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=8, pady=(0, 6))

        # Notebook tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_query = tk.Frame(self.nb)
        self.tab_results = tk.Frame(self.nb)
        self.tab_logs = tk.Frame(self.nb)
        self.tab_history = tk.Frame(self.nb)
        self.tab_settings = tk.Frame(self.nb)

        self.nb.add(self.tab_query, text="Query")
        self.nb.add(self.tab_results, text="Results")
        self.nb.add(self.tab_logs, text="Logs")
        self.nb.add(self.tab_history, text="History")
        self.nb.add(self.tab_settings, text="Settings")

        self._build_tab_query()
        self._build_tab_results()
        self._build_tab_logs()  # Build the logs tab
        self._build_tab_history()
        self._build_tab_settings()

    def _enable_text_copy_paste(self, text_widget: tk.Text):
        # ----- Keyboard shortcuts -----
        text_widget.bind("<Control-c>", lambda e: text_widget.event_generate("<<Copy>>"))
        text_widget.bind("<Control-x>", lambda e: text_widget.event_generate("<<Cut>>"))
        text_widget.bind("<Control-v>", lambda e: text_widget.event_generate("<<Paste>>"))
        text_widget.bind("<Control-a>", self._select_all_text)

        # ----- Right-click context menu -----
        menu = tk.Menu(text_widget, tearoff=0)
        menu.add_command(label="Cut", command=lambda: text_widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: text_widget.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: text_widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Select All", command=lambda: self._select_all_text())

        def show_menu(event):
            menu.tk_popup(event.x_root, event.y_root)

        # Windows / Linux
        text_widget.bind("<Button-3>", show_menu)
        # macOS
        text_widget.bind("<Button-2>", show_menu)

    def _select_all_text(self, event=None):
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            widget.tag_add("sel", "1.0", "end-1c")
            return "break"

    def _build_tab_query(self):
        frm = self.tab_query
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(2, weight=1)

        btns = tk.Frame(frm)
        btns.grid(row=0, column=0, sticky="ew", pady=(6, 10))
        btns.columnconfigure((0, 1, 2, 3), weight=1)

        tk.Button(btns, text="Create Sample CSVs", command=self._create_sample_csvs_threaded)\
            .grid(row=0, column=0, padx=5, sticky="ew")
        tk.Button(btns, text="Build / Upsert Index (Qdrant)", command=self._confirm_and_build_index)\
            .grid(row=0, column=1, padx=5, sticky="ew")
        tk.Button(btns, text="Ping Qdrant", command=self._ping_qdrant_threaded)\
            .grid(row=0, column=2, padx=5, sticky="ew")
        tk.Button(btns, text="Run Query", command=self._run_query_threaded)\
            .grid(row=0, column=3, padx=5, sticky="ew")

        tk.Label(frm, text="Enter your query (Ctrl+Enter to run):", font=("Arial", 12))\
            .grid(row=1, column=0, sticky="w", padx=6)

        self.query_text = tk.Text(frm, height=6, font=("Arial", 12))
        self.query_text.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self._enable_text_copy_paste(self.query_text)        
        self.query_text.bind("<Control-Return>", lambda _e: self._run_query_threaded())

    def _build_tab_results(self):
        frm = self.tab_results
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(frm, font=("Consolas", 11))
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _build_tab_logs(self):
        frm = self.tab_logs
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self.logs_text = scrolledtext.ScrolledText(frm, font=("Consolas", 11))
        self.logs_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _make_scrollable_frame(self, parent: tk.Widget) -> tk.Frame:
        """
        Returns an inner frame inside a Canvas+Scrollbar so we can scroll vertically.
        Put all settings widgets into the returned frame.
        """
        container = tk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Make inner frame width follow canvas width
            canvas.itemconfig(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel support (Windows/macOS/Linux)
        def _on_mousewheel(event):
            # Windows: event.delta is +/-120
            # macOS: event.delta is small
            # Linux: uses Button-4/5
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(_event):
            canvas.yview_scroll(-3, "units")

        def _on_linux_scroll_down(_event):
            canvas.yview_scroll(3, "units")

        # Bind when mouse enters/leaves the scroll area
        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_linux_scroll_up)
            canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Enter>", _bind_wheel)
        inner.bind("<Leave>", _unbind_wheel)

        return inner

    def _build_tab_history(self):
        frm = self.tab_history
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(1, weight=1)

        top = tk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        top.columnconfigure(0, weight=1)

        tk.Label(top, text="Conversation History (from Redis):", font=("Arial", 12)).grid(row=0, column=0, sticky="w")

        btns = tk.Frame(top)
        btns.grid(row=0, column=1, sticky="e")

        tk.Button(btns, text="Refresh", command=self._refresh_history).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Export (History+Logs+Results)", command=self._export_all).pack(side=tk.LEFT, padx=4)

        self.history_text = scrolledtext.ScrolledText(frm, font=("Consolas", 11))
        self.history_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # initial load
        self._refresh_history()

    def _refresh_history(self):
        try:
            self.history_text.delete("1.0", tk.END)

            rs = getattr(self, "redis_store", None)

            # ✅ auto retry init (useful if Redis started after GUI)
            if rs is None:
                try:
                    self._init_redis_store()
                    rs = getattr(self, "redis_store", None)
                except Exception:
                    pass

            if not rs:
                err = getattr(self, "redis_init_error", None)
                if err:
                    self.history_text.insert(tk.END, f"(Redis init failed: {err})\n")
                else:
                    self.history_text.insert(tk.END, "(Redis not initialized)\n")
                return

            if not rs.enabled:
                self.history_text.insert(tk.END, f"(Redis disabled: {rs.last_error or 'unknown'})\n")
                return

            if not rs.ping():
                self.history_text.insert(tk.END, f"(Redis unreachable: {rs.last_error or 'ping failed'})\n")
                return

            # how many history turns to show
            n = 5
            if hasattr(self, "history_turns_var"):
                try:
                    n = int(self.history_turns_var.get())
                except Exception:
                    n = 5
            n = max(1, n)

            turns = rs.get_last_turns(n)
            if not turns:
                self.history_text.insert(tk.END, "(No history yet)\n")
                return

            # show oldest first
            for t in reversed(turns):
                ts = t.get("ts", "")
                q = t.get("q", "")
                a = t.get("a", "")
                self.history_text.insert(tk.END, f"[{ts}]\nUser: {q}\nAssistant: {a}\n\n")

        except Exception as e:
            self.history_text.insert(tk.END, f"[ERROR] refresh_history: {e}\n")

    def _export_to_project(self, kind: str, payload: dict) -> str:
        """Write payload to JSON file inside project folder; return path."""
        exports_dir = self.script_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{kind}_{ts}.json"
        path = exports_dir / fname
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _export_history(self):
        self._export_payload(kind="history")

    def _export_logs(self):
        self._export_payload(kind="logs")

    def _export_results(self):
        self._export_payload(kind="results")

    def _export_payload(self, kind: str):
        try:
            if not getattr(self, "redis_store", None) or not self.redis_store.enabled:
                messagebox.showwarning("Warning", "Redis is disabled/unreachable. Nothing to export.")
                return

            payload_all = self.redis_store.export_all()
            if kind == "history":
                payload = {"meta": payload_all.get("meta", {}), "history": payload_all.get("history", [])}
                default_name = "rag_history_export"
            elif kind == "logs":
                payload = {"meta": payload_all.get("meta", {}), "logs": payload_all.get("logs", [])}
                default_name = "rag_logs_export"
            elif kind == "results":
                payload = {"meta": payload_all.get("meta", {}), "results": payload_all.get("results", [])}
                default_name = "rag_results_export"
            else:
                payload = payload_all
                default_name = "rag_export_all"

            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._export_to_project(kind, payload)
            messagebox.showinfo("Info", f"Exported to: {path}")
            self._append_log(f"[Export] Wrote {kind} export to: {path}")
        except Exception as e:
            self._append_log(f"[ERROR] export_{kind}: {e}")
            messagebox.showwarning("Warning", f"Export failed: {e}")

    def _on_close(self):
        try:
            do_export = messagebox.askyesno("Exit", "Export History/Logs/Results to ./exports before closing?")
            if do_export:
                # Export 3 files automatically under project_folder/exports/
                try:
                    self._export_history()
                    self._export_logs()
                    self._export_results()
                    self._append_log("[Export] Auto-export on close completed.")
                except Exception as e:
                    self._append_log(f"[ERROR] Auto-export on close failed: {e}")
                    messagebox.showwarning("Warning", f"Auto-export failed: {e}")
            self.root.destroy()
        except Exception:
            self.root.destroy()

    def _export_all(self):
        try:
            if not getattr(self, "redis_store", None) or not self.redis_store.enabled:
                messagebox.showwarning("Warning", "Redis is disabled/unreachable. Nothing to export.")
                return
            # Export 3 files
            self._export_history()
            self._export_logs()
            self._export_results()
        except Exception as e:
            self._append_log(f"[ERROR] export_all: {e}")
            messagebox.showwarning("Warning", f"Export failed: {e}")

    def _build_tab_settings(self):
            # frm = self.tab_settings
            frm = self._make_scrollable_frame(self.tab_settings)
            frm.columnconfigure(1, weight=1)

            row = 0

            tk.Label(frm, text="Qdrant URL:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Entry(frm, textvariable=self.qdrant_url).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            row += 1

            tk.Label(frm, text="Collection Prefix:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Entry(frm, textvariable=self.collection_prefix).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            row += 1

            tk.Label(frm, text="Retrieve top_k (Qdrant):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Spinbox(frm, from_=1, to=200, textvariable=self.topk_retrieve, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            tk.Label(frm, text="Use top_k (final contexts):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Spinbox(frm, from_=1, to=20, textvariable=self.topk_use, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            tk.Checkbutton(frm, text="Enable LLM reranking (top_k=10 → best 3)", variable=self.enable_rerank)\
                .grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            tk.Label(frm, text="Text Group Lines (For Indexing Txt Files):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Spinbox(frm, from_=1, to=20, textvariable=self.text_group_lines, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            # # Docs folder
            # tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            # tk.Entry(frm, textvariable=self.docs_dir_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            # row += 1

            # # Docs folder (Entry + Browse button)
            # tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            # docs_row = tk.Frame(frm)
            # docs_row.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            # docs_row.columnconfigure(0, weight=1)
            # tk.Entry(docs_row, textvariable=self.docs_dir_var).grid(row=0, column=0, sticky="ew")
            # tk.Button(docs_row, text="Browse...", command=self._browse_docs_dir).grid(row=0, column=1, padx=(8, 0))
            # row += 1   

            # Docs folder (Entry + Browse + Open)
            tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            docs_row = tk.Frame(frm)
            docs_row.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            docs_row.columnconfigure(0, weight=1)
            tk.Entry(docs_row, textvariable=self.docs_dir_var).grid(row=0, column=0, sticky="ew")
            tk.Button(docs_row, text="Browse...", command=self._browse_docs_dir).grid(row=0, column=1, padx=(8, 0))
            tk.Button(docs_row, text="Open Folder", command=self._open_docs_dir).grid(row=0, column=2, padx=(8, 0))
            row += 1

            # Rerank prompt template
            tk.Label(frm, text="Rerank Prompt Template:").grid(row=row, column=0, sticky="nw", padx=6, pady=6)
            self.rerank_prompt_text = scrolledtext.ScrolledText(frm, height=8, font=("Consolas", 10))
            self.rerank_prompt_text.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            self.rerank_prompt_text.insert("1.0", self.rerank_prompt_var.get())
            row += 1

            # Answer prompt template
            tk.Label(frm, text="Answer Prompt Template:").grid(row=row, column=0, sticky="nw", padx=6, pady=6)
            self.answer_prompt_text = scrolledtext.ScrolledText(frm, height=8, font=("Consolas", 10))
            self.answer_prompt_text.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            self.answer_prompt_text.insert("1.0", self.answer_prompt_var.get())
            row += 1

            # MainProvider
            tk.Label(frm, text="Main LLM Provider:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            ttk.Combobox(
                frm,
                values= ALL_PROVIDERS, #["ollama", "gemini", "openai"],
                textvariable=self.main_provider_var,
                state="readonly",
                width=15
            ).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            # EmbeddingProvider
            tk.Label(frm, text="Embedding LLM Provider:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            ttk.Combobox(
                frm,
                values= ALL_PROVIDERS, #["ollama", "gemini", "openai"],
                textvariable=self.embedding_provider_var,
                state="readonly",
                width=15
            ).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            # Env file path
            tk.Label(frm, text=".env File Path:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Entry(frm, textvariable=self.env_path_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            row += 1

            # 
            ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 6))
            row += 1

            tk.Label(frm, text="Redis URL:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Entry(frm, textvariable=self.redis_url_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            row += 1

            # ---- Data Ids row (single label + 3 fields) ----
            tk.Label(frm, text="Data Ids:").grid(row=row, column=0, sticky="w", padx=6, pady=6)

            ids_frame = tk.Frame(frm)
            ids_frame.grid(row=row, column=1, sticky="w", padx=6, pady=6)

            tk.Label(ids_frame, text="App ID:").grid(row=0, column=0, sticky="w")
            tk.Entry(ids_frame, textvariable=self.app_id_var, width=18)\
                .grid(row=0, column=1, padx=(4, 12), sticky="w")

            tk.Label(ids_frame, text="Session ID:").grid(row=0, column=2, sticky="w")
            tk.Entry(ids_frame, textvariable=self.session_id_var, width=18)\
                .grid(row=0, column=3, padx=(4, 12), sticky="w")

            tk.Label(ids_frame, text="User ID:").grid(row=0, column=4, sticky="w")
            tk.Entry(ids_frame, textvariable=self.user_id_var, width=18)\
                .grid(row=0, column=5, padx=(4, 0), sticky="w")

            row += 1

            tk.Label(frm, text="History turns (for prompt):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
            tk.Spinbox(frm, from_=0, to=50, textvariable=self.history_turns_var, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
            row += 1

            tk.Button(frm, text="Ping Redis", command=self._ping_redis).grid(row=row, column=1, sticky="w", padx=6, pady=(0, 10))
            row += 1

            tk.Button(frm, text="Apply Settings", command=self._apply_settings)\
            #     .grid(row=row, column=1, sticky="w", padx=6)
            
            # tk.Button(frm, text="Reset to Defaults", command=self._reset_to_defaults)\
            #     .grid(row=row, column=2, sticky="w", padx=6)

            # Buttons row (Apply / Reset side-by-side)
            btn_row = tk.Frame(frm)
            btn_row.grid(row=row, column=1, sticky="w", padx=6, pady=(10, 10))

            tk.Button(
                btn_row,
                text="Apply Settings",
                command=self._apply_settings,
                width=16
            ).pack(side=tk.LEFT)

            tk.Button(
                btn_row,
                text="Reset to Defaults",
                command=self._reset_to_defaults,
                width=16
            ).pack(side=tk.LEFT, padx=(10, 0))
            row += 1

        # ---------- Thread safe UI queue ----------
    def _ui_put(self, action: str, payload: Any = None):
        self.ui_queue.put((action, payload))

    def _pump_ui_queue(self):
        try:
            while True:
                action, payload = self.ui_queue.get_nowait()
                if action == "title":
                    self.root.title(str(payload))
                elif action == "log":
                    self._append_log(str(payload))
                elif action == "results":
                    self._set_results(str(payload))
                elif action == "warn":
                    messagebox.showwarning("Warning", str(payload))
                elif action == "info":
                    messagebox.showinfo("Info", str(payload))
        except queue.Empty:
            pass
        self.root.after(50, self._pump_ui_queue)

    def _append_log_for_provider(self):
        env_path_var_text = ""
        if(self.main_provider_var.get() != OLLAMA_PROVIDER):
            env_path_var_text = f", env={self.env_path_var.get()}"
        elif (self.embedding_provider_var.get() != OLLAMA_PROVIDER):
                env_path_var_text = f", env={self.env_path_var.get()}"
        self._append_log(
            f"[Settings] Applied. main_provider={self.main_provider_var.get()}, embedding_provider={self.embedding_provider_var.get()} {env_path_var_text}"
        )

    def _append_log(self, text: str):
        self.logs_text.insert(tk.END, text + "\n")
        try:
            if getattr(self, 'redis_store', None) and self.redis_store.enabled:
                self.redis_store.add_log(text)
        except Exception:
            pass
        self.logs_text.see(tk.END)

    # def _set_results(self, text: str):
    #     self.results_text.delete("1.0", tk.END)
    #     self.results_text.insert(tk.END, text)
    #     self.results_text.see(tk.END)
    #     self.nb.select(self.tab_results)
        try:
            if getattr(self, 'redis_store', None) and self.redis_store.enabled:
                self.redis_store.add_result({'rendered': text})
        except Exception:
            pass

    def _set_results(self, text: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.results_text.insert(tk.END, f"\n{'='*80}\n[{ts}] New Result\n{'='*80}\n")
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.nb.select(self.tab_results)

    # ---------- Settings helper methods ----------
    def _load_settings_file(self) -> Dict[str, Any]:
        return load_settings(self.settings_path)

    def _save_settings_file(self) -> None:
        data = {
            "qdrant_url": self.qdrant_url.get(),
            "collection_prefix": self.collection_prefix.get(),
            "topk_retrieve": int(self.topk_retrieve.get()),
            "topk_use": int(self.topk_use.get()),
            "enable_rerank": bool(self.enable_rerank.get()),
            "text_group_lines": int(self.text_group_lines.get()),
            "docs_dir": self.docs_dir_var.get(),
            "rerank_prompt_template": self.rerank_prompt_var.get(),
            "answer_prompt_template": self.answer_prompt_var.get(),
            "embedding_model": self.embedding_model,
            "main_model": self.main_model,
            "main_provider": self.main_provider_var.get(),
            "embedding_provider": self.embedding_provider_var.get(),
            "env_path": self.env_path_var.get(),
            "redis_url": self.redis_url_var.get(),
            "app_id": self.app_id_var.get(),
            "session_id": self.session_id_var.get(),
            "user_id": self.user_id_var.get(),
            "history_turns": int(self.history_turns_var.get()),
        }
        save_settings(self.settings_path, data)

# ---------- Settings apply ----------
    def _apply_settings(self):
        # Pull latest templates from text widgets
        if hasattr(self, "rerank_prompt_text"):
            self.rerank_prompt_var.set(self.rerank_prompt_text.get("1.0", tk.END).rstrip())
        if hasattr(self, "answer_prompt_text"):
            self.answer_prompt_var.set(self.answer_prompt_text.get("1.0", tk.END).rstrip())

        # Update docs folder path
        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"[Docs] Using folder: {self.csv_dir}")

        self._load_env_from_settings()

        # Recreate engine with new settings and prompts
        self.engine = self._make_engine()
        self._append_log(f"[Settings] Applied. Qdrant={self.qdrant_url.get()}, prefix={self.collection_prefix.get()}, docs_dir={self.csv_dir}")
        
        self._append_log_for_provider()

        # env_path_var_text = ""
        # if(self.provider_var.get() != OLLAMA_PROVIDER):
        #     env_path_var_text = f", env={self.env_path_var.get()}"

        # self._append_log(
        #     f"[Settings] Applied. provider={self.provider_var.get()} {env_path_var_text}"
        # )

        # SAVE to rag_settings.json
        try:
            self._save_settings_file()
            self._append_log(f"[Settings] Saved to: {self.settings_path}")
        except Exception as e:
            self._append_log(f"[ERROR] Failed saving settings: {e}")        

    # def _apply_settings(self):
    #     # recreate engine with new qdrant url/prefix; keep models
    #     self.engine = self._make_engine()
    #     self._append_log(f"[Settings] Applied. Qdrant={self.qdrant_url.get()}, prefix={self.collection_prefix.get()}")

        # re-init Redis
        self._init_redis_store()

    #     # SAVE to rag_settings.json
    #     try:
    #         self._save_settings_file()
    #         self._append_log(f"[Settings] Saved to: {self.settings_path}")
    #     except Exception as e:
    #         self._append_log(f"[ERROR] Failed saving settings: {e}")        

    # ---------- Settings reset ----------
    def _reset_to_defaults(self):
        try:
            defaults = default_settings()

            # Reset TK variables (Settings tab)
            self.qdrant_url.set(defaults["qdrant_url"])
            self.collection_prefix.set(defaults["collection_prefix"])
            self.topk_retrieve.set(int(defaults["topk_retrieve"]))
            self.topk_use.set(int(defaults["topk_use"]))
            self.enable_rerank.set(bool(defaults["enable_rerank"]))
            self.text_group_lines.set(int(defaults["text_group_lines"]))

            # Reset models too (optional but consistent with saved file)
            self.embedding_model = defaults["embedding_model"]
            self.main_model = defaults["main_model"]

            # Reflect in UI combos if built
            if hasattr(self, "embedding_combo"):
                self.embedding_combo.set(self.embedding_model)
            if hasattr(self, "main_combo"):
                self.main_combo.set(self.main_model)

            # Reset prompt templates and docs dir
            self.docs_dir_var.set(defaults["docs_dir"])
            self.rerank_prompt_var.set(defaults["rerank_prompt_template"])
            self.answer_prompt_var.set(defaults["answer_prompt_template"])

            # Reflect in text widgets if built
            if hasattr(self, "rerank_prompt_text"):
                self.rerank_prompt_text.delete("1.0", tk.END)
                self.rerank_prompt_text.insert("1.0", self.rerank_prompt_var.get())
            if hasattr(self, "answer_prompt_text"):
                self.answer_prompt_text.delete("1.0", tk.END)
                self.answer_prompt_text.insert("1.0", self.answer_prompt_var.get())

            self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
            self.csv_dir.mkdir(parents=True, exist_ok=True)

            # Apply (recreate engine) + set models
            self.engine = self._make_engine()
            self.engine.set_models(self.embedding_model, self.main_model)

            # Save immediately
            self._save_settings_file()

            self._append_log("[Settings] Reset to defaults and saved.")
            messagebox.showinfo("Info", "Settings reset to defaults.")
        except Exception as e:
            self._append_log(f"[ERROR] reset_to_defaults: {e}")
            messagebox.showwarning("Warning", f"Reset failed: {e}")

    # ---------- Model selection handlers ----------
    def _on_embedding_changed(self, _event):
        self.embedding_model = self.embedding_combo.get()
        self.engine.set_models(self.embedding_model, self.main_model)
        self._append_log(f"[Models] Embedding model set to: {self.embedding_model}")

        self.embedding_provider_var.set(OLLAMA_PROVIDER)
        if(self.embedding_model in EMBEDDING_MODELS_GEMINI):
            self.embedding_provider_var.set(GEMINI_PROVIDER)
        if(self.embedding_model in EMBEDDING_MODELS_OPENAI):
                    self.embedding_provider_var.set(OPENAI_PROVIDER)            
        self._append_log_for_provider()

        try:
            self._save_settings_file()
        except Exception:
            pass

    def _on_main_changed(self, _event):
        self.main_model = self.main_combo.get()
        self.engine.set_models(self.embedding_model, self.main_model)
        self._append_log(f"[Models] Main model set to: {self.main_model}")

        self.main_provider_var.set(OLLAMA_PROVIDER)
        if(self.main_model in CHAT_MODELS_GEMINI):
            self.main_provider_var.set(GEMINI_PROVIDER)
        if(self.main_model in CHAT_MODELS_OPENAI):
            self.main_provider_var.set(OPENAI_PROVIDER)
        self._append_log_for_provider()

        try:
            self._save_settings_file()
        except Exception:
            pass

    # ---------- Worker thread wrappers ----------
    def _create_sample_csvs_threaded(self):
        threading.Thread(target=self._create_sample_csvs, daemon=True).start()

    def _confirm_and_build_index(self):
        proceed = messagebox.askyesno(
            "Confirm Index Build",
            "This will (re)build the index and may overwrite existing data.\n\n"
            "Do you want to continue?"
        )
        if not proceed:
            self._append_log("[Index] Build canceled by user.")
            return

        self._build_index_threaded()

    def _build_index_threaded(self):
        threading.Thread(target=self._build_index, daemon=True).start()

    def _ping_redis(self):
        # UI thread
        self._init_redis_store()
        if getattr(self, "redis_store", None) and self.redis_store.enabled and self.redis_store.ping():
            messagebox.showinfo("Info", "Redis is reachable.")
        else:
            reason = None
            if getattr(self, "redis_store", None):
                reason = self.redis_store.last_error
            messagebox.showwarning("Warning", f"Redis not reachable. {reason or ''}".strip())

    def _ping_qdrant_threaded(self):
            threading.Thread(target=self._ping_qdrant, daemon=True).start()

    def _run_query_threaded(self):
        threading.Thread(target=self._run_query, daemon=True).start()

    # ---------- Workers ----------
    def _create_sample_csvs(self):
        self._ui_put("title", APP_TITLE + " : create_sample_csvs ...")
        try:
            self.csv_dir.mkdir(parents=True, exist_ok=True)

            customer_data = [
                {"name": "Alice Johnson", "orders": "Laptop, Mouse", "description": "Customer interested in tech gadgets"},
                {"name": "Bob Smith", "orders": "Keyboard, Monitor", "description": "Customer upgrading home office"},
                {"name": "Charlie Davis", "orders": "Smartphone, Tablet", "description": "Customer seeking mobile devices"},
                {"name": "Diana Garcia", "orders": "Headphones, Speakers", "description": "Customer focused on audio equipment"},
            ]
            pd.DataFrame(customer_data).to_csv(self.csv_dir / "customer_data.csv", index=False)

            medical_data = [
                {"ActivityCode": "00100", "AcceptedDiagnosis": "C07, C08.0, C08.1, C08.9, C79.89, C79.9", "Rule": "MedicalNecessity"},
                {"ActivityCode": "00102", "AcceptedDiagnosis": "Q36.0, Q36.1, Q36.9, Q37.0, Q37.1, Q37.2", "Rule": "MedicalNecessity"},
                {"ActivityCode": "00103", "AcceptedDiagnosis": "H02.121, H02.122, H02.123, H02.124", "Rule": "VPSActivityFound"},
                {"ActivityCode": "00104", "AcceptedDiagnosis": "H02.131, H02.132, H02.133, H02.134", "Rule": "VPSActivityFound"},
            ]
            pd.DataFrame(medical_data).to_csv(self.csv_dir / "medical_rules.csv", index=False)

            self._ui_put("log", f"[Data] Sample CSVs created in: {self.csv_dir}")
        except Exception as e:
            self._ui_put("log", f"[ERROR] create_sample_csvs: {e}")
        finally:
            self._ui_put("title", APP_TITLE)

    def _ping_qdrant(self):
        self._ui_put("title", APP_TITLE + " : ping_qdrant ...")
        try:
            # simple ping: list collections
            cols = self.engine.client.get_collections()
            self._ui_put("log", f"[Qdrant] Connected OK. Collections: {len(cols.collections)}")
            self._ui_put("info", "Qdrant is reachable.")
        except Exception as e:
            self._ui_put("log", f"[ERROR] Qdrant ping failed: {e}")
            self._ui_put("warn", f"Qdrant not reachable at {self.qdrant_url.get()}\nStart Qdrant and try again.")
        finally:
            self._ui_put("title", APP_TITLE)

    def _build_index(self):
        self._ui_put("title", APP_TITLE + " : build_index (Qdrant) ...")
        try:
            # refresh engine settings (if user changed URL/prefix but didn't apply)
            self.engine = self._make_engine()

            # total = self.engine.build_from_csv_folder(self.csv_dir)
            total = self.engine.build_from_folder(
                self.csv_dir,
                txt_chunk_chars=900,
                txt_overlap=120,
                group_lines=self.text_group_lines.get(),
            )
            if total == 0:
                self._ui_put("log", "[Index] No chunks found in CSV folder.")
            else:
                self._ui_put("log", f"[Index] Upserted {total} chunks into: {self.engine.collection_name}")
        except Exception as e:
            self._ui_put("log", f"[ERROR] build_index: {e}")
            self._ui_put("warn", "Index build failed. Check Qdrant + Ollama are running.")
        finally:
            self._ui_put("title", APP_TITLE)

    def _run_query(self):
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            self._ui_put("warn", "Please enter a query.")
            return

        self._ui_put("title", APP_TITLE + " : run_query ...")
        try:
            # refresh engine settings
            self.engine = self._make_engine()

            topk_retrieve = int(self.topk_retrieve.get())
            topk_use = int(self.topk_use.get())
            enable_rerank = bool(self.enable_rerank.get())

            history_block = ""
            if getattr(self, 'redis_store', None) and self.redis_store.enabled:
                # turns = self.redis_store.get_last_turns(int(self.settings.get('history_turns', 5)))
                turns = self.redis_store.get_last_turns(self.history_turns_var.get())
                history_block = format_history_for_prompt(turns)
            original_query = query
            result, query  = self.engine.answer(
                query,
                history_block=history_block,
                top_k_retrieve=topk_retrieve,
                top_k_use=topk_use,
                enable_rerank=enable_rerank,
            )

            hits = result.get("hits", [])
            selected = result.get("selected", [])
            context = result.get("context", "")
            answer = result.get("answer", "")
            latency = float(result.get("latency", 0.0))
            collection = result.get("collection", "")

            try:
                if getattr(self, "redis_store", None) and self.redis_store.enabled:
                    self.redis_store.append_turn(query, answer, meta={"collection": collection})
            except Exception:
                pass

            # refresh History tab view
            try:
                if hasattr(self, "history_text"):
                    self._refresh_history()
            except Exception:
                pass

            # Build a clean, production-style report   orginal_query if query != orginal_query else c
            lines = []
            # lines.append(f"Query: {query} {'' if query == orginal_query else f'/n(modified from original:/n {orginal_query})'}")
            # lines.append(
            #     f"Query: {query}" 
            #     f"{'' if query == orginal_query else f'\n(modified from original:\n{orginal_query})'}"
            # )
            lines.append(
                "".join([
                    f"Query: {query}",
                    "" if query == original_query else
                    f"\n(modified from original:\n{original_query})"
                ])
            )            
            lines.append("")
            lines.append("Answer:")
            lines.append(answer)
            lines.append("")            
            lines.append(f"Collection: {collection}")
            lines.append(f"Provider: main={self.main_provider_var.get()} | embedding={self.embedding_provider_var.get()}")
            lines.append(f"Models: main={self.main_model} | embedding={self.embedding_model}")
            lines.append(f"Retrieve top_k={topk_retrieve} | Use top_k={topk_use} | Rerank={enable_rerank} | history_used={self.history_turns_var.get()}")
            lines.append(f"Ids: App_id={self.app_id_var.get()} | Session_id={self.session_id_var.get()} | User_id={self.user_id_var.get()}")
            lines.append(f"Latency: {latency:.3f} seconds")
            lines.append("")
            lines.append("Selected Context:")
            lines.append(context or "(empty)")
            lines.append("")
            lines.append("history_block:")
            lines.append(history_block or "(empty)")
            # lines.append("")            
            # lines.append("Answer:")
            # lines.append(answer)

            # Show all hits in logs for debugging
            self._ui_put("log", "\n---\n".join([
                f"[Query] {query}",
                f"[RAG] collection={collection} retrieve={topk_retrieve} use={topk_use} rerank={enable_rerank}",
                f"[RAG] selected_indices={selected}",
                f"[RAG] hits_count={len(hits)} latency={latency:.3f}s history_used={self.history_turns_var.get()}",
            ]))

            self._ui_put("results", "\n".join(lines))

        except Exception as e:
            self._ui_put("log", f"[ERROR] run_query: {e}")
            self._ui_put("warn", f"Query failed: {e}")
        finally:
            self._ui_put("title", APP_TITLE)


def run() -> None:
    root = tk.Tk()
    RAGAppGUI(root)
    root.mainloop()
