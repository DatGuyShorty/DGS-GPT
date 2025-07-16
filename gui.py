import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog, simpledialog
import threading
import torch
import os
import queue
import sys

# Matplotlib for plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import necessary components from ShitGPT
try:
    from ShitGPT import (
        Config,
        GPTLanguageModel,
        Trainer,
        load_config,
        run_hyperparameter_optimization,
        encode,
        decode,
        device,
        vocab_size
    )
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import from ShitGPT.py: {e}\nMake sure both files are in the same directory.")
    exit()

# A custom stream object to redirect stdout
class QueueStream(object):
    def __init__(self, queue):
        self.queue = queue
    def write(self, text):
        self.queue.put(text)
    def flush(self):
        pass

class GPT_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ShitGPT GUI")
        self.root.geometry("800x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.loss_queue = queue.Queue()
        self.losses = []
        self.stop_training_event = None
        
        self.config = load_config()
        self.config.vocab_size = vocab_size # Set correct vocab size
        self.trainer = Trainer(self.config, loss_callback=self.queue_loss)

        # Create status bar first, as other methods may use it.
        self.status_text = tk.StringVar()
        self.status_bar = tk.Label(root, textvariable=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.model_loaded = self.load_model_weights()
        self.status_text.set("Ready." if self.model_loaded else "gpt_model.pth not found. Using a random model.")
        
        self.create_menu()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.create_generation_tab()
        self.create_training_tab()

        self.root.after(100, self.process_loss_queue)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)

        actions_menu = tk.Menu(menubar, tearoff=0)
        actions_menu.add_command(label="Run Hyperparameter Optimization", command=self.run_hyperparam_optimization_action)
        menubar.add_cascade(label="Actions", menu=actions_menu)

        self.root.config(menu=menubar)

    def save_model(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            title="Save Model As"
        )
        if filepath:
            self.trainer.save_model(filepath)
            self.status_text.set(f"Model saved to {os.path.basename(filepath)}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            title="Load Model"
        )
        if filepath and os.path.exists(filepath):
            try:
                # Use strict=False to allow loading partial or different models
                self.trainer.model.load_state_dict(torch.load(filepath, map_location=device), strict=False)
                self.status_text.set(f"Model loaded from {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model weights:\n{e}")
                self.status_text.set("Failed to load model.")
        elif filepath:
            messagebox.showerror("File Not Found", f"The file '{filepath}' does not exist.")

    def run_hyperparam_optimization_action(self):
        n_trials = simpledialog.askinteger("Optuna", "How many optimization trials to run?", initialvalue=10, minvalue=1)
        if n_trials is None: return
        steps_per_trial = simpledialog.askinteger("Optuna", "How many training steps per trial?", initialvalue=100, minvalue=1)
        if steps_per_trial is None: return

        log_window = tk.Toplevel(self.root)
        log_window.title("Hyperparameter Optimization Log")
        log_window.geometry("800x600")
        log_text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, state='disabled')
        log_text.pack(expand=True, fill='both')

        thread = threading.Thread(
            target=self.run_optimization_thread,
            args=(n_trials, steps_per_trial, log_window, log_text)
        )
        thread.daemon = True
        thread.start()

    def run_optimization_thread(self, n_trials, steps_per_trial, log_window, log_text):
        log_queue = queue.Queue()
        original_stdout = sys.stdout
        sys.stdout = QueueStream(log_queue)

        def update_log():
            while not log_queue.empty():
                line = log_queue.get_nowait()
                log_text.config(state='normal')
                log_text.insert(tk.END, line)
                log_text.config(state='disabled')
                log_text.see(tk.END)
            log_window.after(100, update_log)

        log_window.after(100, update_log)

        try:
            new_config = run_hyperparameter_optimization(self.config, n_trials, steps_per_trial)
            self.root.after(0, self.on_optimization_complete, new_config)
        except Exception as e:
            print(f"\n--- OPTIMIZATION FAILED ---\n{e}")
        finally:
            sys.stdout = original_stdout
            print("Optimization process finished.")

    def on_optimization_complete(self, new_config):
        messagebox.showinfo("Optimization Complete", "Hyperparameter optimization finished. The model has been updated with the best parameters found.")
        self.config = new_config
        self.config.vocab_size = vocab_size # Also update vocab size on new config
        self.trainer = Trainer(self.config, loss_callback=self.queue_loss)
        self.status_text.set("Model updated with new hyperparameters.")

    def on_closing(self):
        self.root.destroy()

    def load_model_weights(self):
        model_path = "gpt_model.pth"
        if os.path.exists(model_path):
            try:
                # Use strict=False to allow loading partial or different models
                self.trainer.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                self.status_text.set("Model loaded successfully.")
                return True
            except Exception as e:
                messagebox.showwarning("Model Load Error", f"Could not load model weights from {model_path}:\n{e}\n\nUsing a randomly initialized model.")
        else:
            messagebox.showwarning("Model Not Found", f"Model file '{model_path}' not found.\nThe model will have random weights and produce gibberish.")
        return False

    def create_generation_tab(self):
        gen_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(gen_frame, text='Text Generation')

        tk.Label(gen_frame, text="Enter your prompt:").pack(pady=5, anchor="w")
        self.prompt_input = scrolledtext.ScrolledText(gen_frame, height=8, width=80, wrap=tk.WORD)
        self.prompt_input.pack(pady=5, fill="x", expand=True)

        self.generate_button = tk.Button(gen_frame, text="Generate Text", command=self.start_generation_thread)
        self.generate_button.pack(pady=10)

        tk.Label(gen_frame, text="Generated Text:").pack(pady=5, anchor="w")
        self.output_text = scrolledtext.ScrolledText(gen_frame, height=15, width=80, wrap=tk.WORD, state="disabled")
        self.output_text.pack(pady=5, fill="both", expand=True)

    def create_training_tab(self):
        train_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(train_frame, text='Training')

        controls_frame = tk.Frame(train_frame)
        controls_frame.pack(fill='x', pady=5)
        tk.Label(controls_frame, text="Iterations:").pack(side='left', padx=5)
        self.iters_input = tk.Entry(controls_frame, width=10)
        self.iters_input.insert(0, "1000")
        self.iters_input.pack(side='left', padx=5)

        tk.Label(controls_frame, text="Plot Step:").pack(side='left', padx=5)
        self.plot_step_input = tk.Entry(controls_frame, width=10)
        self.plot_step_input.insert(0, "100")
        self.plot_step_input.pack(side='left', padx=5)

        self.train_button = tk.Button(controls_frame, text="Start Training", command=self.start_training_thread)
        self.train_button.pack(side='left', padx=5)

        self.stop_button = tk.Button(controls_frame, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.pack(side='left', padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(train_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Update Step")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], 'r-')

        self.canvas = FigureCanvasTkAgg(self.fig, master=train_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def queue_loss(self, loss):
        self.loss_queue.put(loss)

    def process_loss_queue(self):
        try:
            while not self.loss_queue.empty():
                loss = self.loss_queue.get_nowait()
                self.losses.append(loss)
                self.update_plot()
        finally:
            self.root.after(100, self.process_loss_queue)

    def update_plot(self):
        self.line.set_data(range(len(self.losses)), self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()

    def start_training_thread(self):
        try:
            max_iters = int(self.iters_input.get())
            plot_step_size = int(self.plot_step_input.get())
            if max_iters <= 0 or plot_step_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive integers for iterations and plot step.")
            return

        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.generate_button.config(state="disabled")
        self.status_text.set(f"Training for {max_iters} iterations...")
        self.losses = []
        self.update_plot()

        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = max_iters

        self.stop_training_event = threading.Event()
        thread = threading.Thread(target=self.run_training, args=(max_iters, plot_step_size))
        thread.daemon = True
        thread.start()

    def run_training(self, max_iters, plot_step_size):
        try:
            self.trainer.train(max_iters, plot_step_size, self.stop_training_event, self.update_progress)
            if self.stop_training_event.is_set():
                self.root.after(0, lambda: self.status_text.set("Training stopped by user."))
            else:
                self.root.after(0, lambda: self.status_text.set("Training complete. Ready."))
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set(f"Training error: {e}"))
        finally:
            self.root.after(0, lambda: self.train_button.config(state="normal"))
            self.root.after(0, lambda: self.stop_button.config(state="disabled"))
            self.root.after(0, lambda: self.generate_button.config(state="normal"))

    def stop_training(self):
        if self.stop_training_event:
            self.stop_training_event.set()
        self.stop_button.config(state="disabled")

    def update_progress(self, current_step):
        self.root.after(0, self._set_progress, current_step)

    def _set_progress(self, value):
        self.progress_bar['value'] = value

    def start_generation_thread(self):
        self.generate_button.config(state="disabled")
        self.train_button.config(state="disabled")
        self.status_text.set("Generating...")
        thread = threading.Thread(target=self.generate_text)
        thread.daemon = True
        thread.start()

    def generate_text(self):
        prompt = self.prompt_input.get("1.0", tk.END).strip()
        if not prompt:
            self.root.after(0, lambda: self.status_text.set("Please enter a prompt."))
            self.root.after(0, lambda: self.generate_button.config(state="normal"))
            self.root.after(0, lambda: self.train_button.config(state="normal"))
            return

        try:
            with torch.no_grad():
                self.trainer.model.eval()
                idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                gen_ids = self.trainer.model.generate(idx, max_new_tokens=200)
                generated_text = decode(gen_ids[0].tolist())
                self.trainer.model.train()

            self.root.after(0, self.update_output, generated_text)
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set(f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self.generate_button.config(state="normal"))
            self.root.after(0, lambda: self.train_button.config(state="normal"))

    def update_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")
        self.status_text.set("Generation complete. Ready.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPT_GUI(root)
    root.mainloop()
