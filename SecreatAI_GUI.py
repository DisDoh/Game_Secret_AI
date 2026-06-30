#copyright Reda Benjamin Meyer

import contextlib
import io
import os
import pickle
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import Decode_only
import Encode_only
import Training_only


APP_DIR = Path(__file__).resolve().parent
DECODED_DIR = APP_DIR / "decoded"
MODEL_PATTERN = "*.pkl"


class LogWriter(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        if text:
            self.callback(text)
        return len(text)

    def flush(self):
        pass


class SecreatAIGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SecreatAI")
        self.geometry("820x560")
        self.minsize(720, 480)

        self.selected_file = tk.StringVar()
        self.model_file = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.training_active = False
        self.training_stop_event = threading.Event()

        self._build_ui()
        self.refresh_models()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        header = ttk.Frame(self, padding=(16, 14, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text="SecreatAI Encoder / Decoder", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(self, padding=(16, 8))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_file, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew", pady=6)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_graph())
        ttk.Button(controls, text="New Model", command=self.create_new_model).grid(row=0, column=2, padx=(8, 0), pady=6)
        ttk.Button(controls, text="Refresh", command=self.refresh_models).grid(row=0, column=3, padx=(8, 0), pady=6)

        ttk.Label(controls, text="File").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(controls, textvariable=self.selected_file).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(controls, text="Browse", command=self.pick_file).grid(row=1, column=2, columnspan=2,
                                                                         sticky="ew", padx=(8, 0), pady=6)

        actions = ttk.Frame(self, padding=(16, 8))
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(5, weight=1)

        self.encode_button = ttk.Button(actions, text="Encode", command=self.encode_file)
        self.encode_button.grid(row=0, column=0, padx=(0, 8))
        self.decode_button = ttk.Button(actions, text="Decode", command=self.decode_file)
        self.decode_button.grid(row=0, column=1, padx=(0, 8))
        self.train_button = ttk.Button(actions, text="Train / Resume", command=self.train_model)
        self.train_button.grid(row=0, column=2, padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_training, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=(0, 8))
        ttk.Button(actions, text="Clear Log", command=self.clear_log).grid(row=0, column=4, padx=(0, 8))
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=5, sticky="e")

        main_pane = ttk.PanedWindow(self, orient="horizontal")
        main_pane.grid(row=3, column=0, sticky="nsew", padx=16, pady=(8, 16))

        log_frame = ttk.Frame(main_pane)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_pane.add(log_frame, weight=3)

        self.log = tk.Text(log_frame, wrap="word", height=18, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        graph_frame = ttk.Frame(main_pane)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        main_pane.add(graph_frame, weight=2)

        self.figure = Figure(figsize=(4.2, 3.2), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Training Loss")
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel("Loss")
        self.axis.grid(True, alpha=0.25)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def refresh_models(self):
        models = sorted(path.name for path in APP_DIR.glob(MODEL_PATTERN))
        self.model_combo.configure(values=models)
        if models and self.model_file.get() not in models:
            self.model_file.set("model.pkl" if "model.pkl" in models else models[0])
        elif not models:
            self.model_file.set("")
        self.write_log(f"Found {len(models)} model file(s).\n")
        self.update_graph()

    def create_new_model(self):
        raw_name = simpledialog.askstring("New model", "Model name:", parent=self)
        if not raw_name:
            return

        model_name = self.normalize_model_name(raw_name)
        model_path = APP_DIR / model_name
        if model_path.exists():
            use_existing = messagebox.askyesno("Model exists", f"{model_name} already exists. Select it?")
            if not use_existing:
                return

        current_values = list(self.model_combo.cget("values"))
        if model_name not in current_values:
            current_values.append(model_name)
            self.model_combo.configure(values=sorted(current_values))

        self.model_file.set(model_name)
        self.update_graph()
        if model_path.exists():
            self.write_log(f"Selected existing model: {model_name}\n")
        else:
            self.write_log(f"New model selected: {model_name}. Press Train / Resume to create it.\n")

    def normalize_model_name(self, raw_name):
        model_name = Path(raw_name.strip()).name
        if not model_name.endswith(".pkl"):
            model_name = f"{model_name}.pkl"
        return model_name

    def pick_file(self):
        filename = filedialog.askopenfilename(initialdir=APP_DIR)
        if filename:
            self.selected_file.set(filename)

    def encode_file(self):
        self._run_with_file("encode")

    def decode_file(self):
        self._run_with_file("decode")

    def train_model(self):
        model_name = self.model_file.get() or "model.pkl"
        if not self.model_file.get():
            self.model_file.set(model_name)

        def task():
            Training_only.model_name = model_name
            Training_only.main(stop_event=self.training_stop_event)

        self.training_stop_event.clear()
        self.training_active = True
        self.schedule_graph_update()
        self._run_task("Training", task, model_name=model_name)

    def stop_training(self):
        if not self.training_active:
            return
        self.training_stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping training...")
        self.write_log("\nStop requested. Training will stop after the current epoch is saved.\n")

    def _run_with_file(self, mode):
        model_name = self.model_file.get()
        source = self.selected_file.get()
        if not model_name:
            messagebox.showerror("Missing model", "Choose a model file first.")
            return
        if not source:
            messagebox.showerror("Missing file", "Choose a file first.")
            return
        source_path = Path(source)
        if not source_path.exists():
            messagebox.showerror("File not found", str(source_path))
            return

        def task():
            working_file = self.copy_to_app_dir(source_path)
            if mode == "encode":
                output = APP_DIR / f"{working_file.name}.aiz"
                previous_mtime = output.stat().st_mtime if output.exists() else None
                accuracy_passed = Encode_only.main(model_name, working_file.name)
                current_mtime = output.stat().st_mtime if output.exists() else None
                if accuracy_passed and output.exists() and current_mtime != previous_mtime:
                    self.write_log(f"\nEncoded file: {output}\n")
                elif accuracy_passed and output.exists():
                    self.write_log(f"\nEncoded file already exists: {output}\n")
                else:
                    self.write_log("\nEncode stopped because the accuracy test did not pass.\n")
            else:
                Decode_only.main(model_name, working_file.name)
                self.write_log(f"\nDecoded files are in: {DECODED_DIR}\n")

        self._run_task(mode.capitalize(), task, model_name=model_name)

    def copy_to_app_dir(self, source_path):
        destination = APP_DIR / source_path.name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
            self.write_log(f"Copied input to: {destination}\n")
        return destination

    def _run_task(self, label, task, model_name=None):
        self.set_busy(True, f"{label}...")
        self.write_log(f"\n--- {label} started ---\n")
        if model_name:
            self.write_log(f"Model: {model_name}\n")

        def worker():
            old_cwd = Path.cwd()
            try:
                os.chdir(APP_DIR)
                with contextlib.redirect_stdout(LogWriter(self.write_log_threadsafe)):
                    task()
            except Exception as exc:
                self.write_log_threadsafe(f"\nERROR: {exc}\n")
                self.after(0, lambda: messagebox.showerror(f"{label} failed", str(exc)))
            finally:
                os.chdir(old_cwd)
                self.write_log_threadsafe(f"--- {label} finished ---\n")
                if label == "Training":
                    self.training_active = False
                self.after(0, self.refresh_models)
                self.after(0, lambda: self.set_busy(False, "Ready"))

        threading.Thread(target=worker, daemon=True).start()

    def set_busy(self, busy, status):
        state = "disabled" if busy else "normal"
        self.encode_button.configure(state=state)
        self.decode_button.configure(state=state)
        self.train_button.configure(state=state)
        self.stop_button.configure(state="normal" if busy and self.training_active else "disabled")
        self.status.set(status)

    def write_log_threadsafe(self, text):
        self.after(0, lambda: self.write_log(text))

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def schedule_graph_update(self):
        self.update_graph()
        if self.training_active:
            self.after(2000, self.schedule_graph_update)

    def update_graph(self):
        model_path = APP_DIR / self.model_file.get()
        train_losses, val_losses = self.load_losses(model_path)
        paired_count = min(len(train_losses), len(val_losses))
        epochs = range(1, paired_count + 1)

        self.axis.clear()
        self.axis.set_title("Training and Validation Loss")
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel("Loss")
        self.axis.grid(True, alpha=0.25)

        if paired_count:
            self.axis.plot(epochs, train_losses[:paired_count], label="Training")
            self.axis.plot(epochs, val_losses[:paired_count], label="Validation")
            self.axis.legend(loc="best")
        else:
            self.axis.text(0.5, 0.5, "No training history yet", ha="center", va="center",
                           transform=self.axis.transAxes)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def load_losses(self, model_path):
        if not model_path.exists() or not model_path.is_file():
            return [], []
        try:
            with model_path.open("rb") as file:
                saved_data = pickle.load(file)
            model = saved_data.get("model", {})
            train_losses = list(model.get("train_losses", []))
            val_losses = list(model.get("val_losses", []))
            return train_losses, val_losses
        except Exception:
            return [], []

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    app = SecreatAIGUI()
    app.mainloop()
