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
    from version import __version__, get_version_info
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
        self.root.title(f"DGS-GPT v{__version__}")
        self.root.geometry("800x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.loss_queue = queue.Queue()
        self.losses = []
        self.stop_training_event = None
        self.optim_queue = None
        self.stop_optimization_event = None
        
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
        self.create_hyperparameter_tab()

        self.root.after(100, self.process_loss_queue)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

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

    def start_optimization_thread(self):
        try:
            n_trials = int(self.trials_input.get())
            steps_per_trial = int(self.steps_per_trial_input.get())
            if n_trials <= 0 or steps_per_trial <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive integers for trials and steps per trial.")
            return

        self.optimize_button.config(state="disabled")
        self.stop_optimize_button.config(state="normal")
        self.train_button.config(state="disabled")
        self.generate_button.config(state="disabled")
        self.status_text.set(f"Running hyperparameter optimization with {n_trials} trials...")
        
        # Clear log
        self.optim_log_text.config(state="normal")
        self.optim_log_text.delete("1.0", tk.END)
        self.optim_log_text.config(state="disabled")

        self.optim_progress_bar['value'] = 0
        self.optim_progress_bar['maximum'] = n_trials

        self.optim_queue = queue.Queue()
        self.stop_optimization_event = threading.Event()
        
        thread = threading.Thread(target=self.run_optimization_thread, args=(n_trials, steps_per_trial))
        thread.daemon = True
        thread.start()
        
        # Start updating the log display
        self.root.after(100, self.process_optim_queue)

    def run_optimization_thread(self, n_trials, steps_per_trial):
        original_stdout = sys.stdout
        sys.stdout = QueueStream(self.optim_queue)

        try:
            new_config = run_hyperparameter_optimization(self.config, n_trials, steps_per_trial)
            self.root.after(0, self.on_optimization_complete, new_config)
        except Exception as e:
            print(f"\n--- OPTIMIZATION FAILED ---\n{e}")
            self.root.after(0, lambda: self.status_text.set(f"Optimization error: {e}"))
        finally:
            sys.stdout = original_stdout
            self.root.after(0, self.optimization_finished)

    def process_optim_queue(self):
        updated = False
        try:
            while not self.optim_queue.empty():
                line = self.optim_queue.get_nowait()
                self.optim_log_text.config(state='normal')
                self.optim_log_text.insert(tk.END, line)
                self.optim_log_text.config(state='disabled')
                self.optim_log_text.see(tk.END)
                updated = True
        except:
            pass
        
        if updated:
            self.root.update_idletasks()
        
        # Continue processing if optimization is running
        if self.optimize_button['state'] == 'disabled':
            self.root.after(100, self.process_optim_queue)

    def stop_optimization(self):
        if hasattr(self, 'stop_optimization_event'):
            self.stop_optimization_event.set()
        self.stop_optimize_button.config(state="disabled")
        self.status_text.set("Stopping optimization...")

    def optimization_finished(self):
        self.optimize_button.config(state="normal")
        self.stop_optimize_button.config(state="disabled")
        self.train_button.config(state="normal")
        self.generate_button.config(state="normal")

    def on_optimization_complete(self, new_config):
        messagebox.showinfo("Optimization Complete", "Hyperparameter optimization finished. The model has been updated with the best parameters found.")
        self.config = new_config
        self.config.vocab_size = vocab_size # Also update vocab size on new config
        self.trainer = Trainer(self.config, loss_callback=self.queue_loss)
        self.update_hyperparameter_display()  # Update the display with new hyperparameters
        self.status_text.set("Model updated with new hyperparameters.")

    def show_about(self):
        """Show about dialog with version information"""
        version_info = get_version_info()
        about_text = f"""DGS-GPT v{version_info['version']}

{version_info['description']}

Author: {version_info['author']}
License: {version_info['license']}

A modern GPT implementation featuring:
• Advanced transformer architecture
• Multi-head and grouped query attention
• Mixture of Experts (MoE) support
• Real-time training visualization
• Hyperparameter optimization with Optuna
• Cross-platform compatibility

For more information, visit:
https://github.com/DatGuyShorty/DGS-GPT"""
        
        messagebox.showinfo("About DGS-GPT", about_text)

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

        # Current hyperparameters display
        hyperparam_display_frame = tk.LabelFrame(train_frame, text="Current Hyperparameters", padx=10, pady=5)
        hyperparam_display_frame.pack(fill='x', pady=5)
        
        self.hyperparam_text = scrolledtext.ScrolledText(hyperparam_display_frame, height=8, wrap=tk.WORD, state="disabled")
        self.hyperparam_text.pack(fill='x', pady=5)
        self.update_hyperparameter_display()

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Update Step")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], 'r-')

        self.canvas = FigureCanvasTkAgg(self.fig, master=train_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_hyperparameter_tab(self):
        hyperparam_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(hyperparam_frame, text='Hyperparameter Optimization')

        # Controls frame
        controls_frame = tk.Frame(hyperparam_frame)
        controls_frame.pack(fill='x', pady=5)
        
        tk.Label(controls_frame, text="Number of Trials:").pack(side='left', padx=5)
        self.trials_input = tk.Entry(controls_frame, width=10)
        self.trials_input.insert(0, "10")
        self.trials_input.pack(side='left', padx=5)

        tk.Label(controls_frame, text="Steps per Trial:").pack(side='left', padx=5)
        self.steps_per_trial_input = tk.Entry(controls_frame, width=10)
        self.steps_per_trial_input.insert(0, "100")
        self.steps_per_trial_input.pack(side='left', padx=5)

        self.optimize_button = tk.Button(controls_frame, text="Start Optimization", command=self.start_optimization_thread)
        self.optimize_button.pack(side='left', padx=5)

        self.stop_optimize_button = tk.Button(controls_frame, text="Stop Optimization", command=self.stop_optimization, state="disabled")
        self.stop_optimize_button.pack(side='left', padx=5)

        # Progress bar for optimization
        self.optim_progress_bar = ttk.Progressbar(hyperparam_frame, orient='horizontal', mode='determinate')
        self.optim_progress_bar.pack(fill='x', pady=5)

        # Log display
        tk.Label(hyperparam_frame, text="Optimization Log:").pack(pady=(10,5), anchor="w")
        self.optim_log_text = scrolledtext.ScrolledText(hyperparam_frame, height=20, wrap=tk.WORD, state="disabled")
        self.optim_log_text.pack(fill="both", expand=True, pady=5)

    def update_hyperparameter_display(self):
        """Update the hyperparameter display with current config values"""
        config_text = f"""Learning Rate: {self.config.learning_rate}
Weight Decay: {self.config.weight_decay}
Embedding Dimension: {self.config.n_embd}
Number of Layers: {self.config.n_layer}
Number of Heads: {self.config.n_head}
Block Size: {self.config.block_size}
Batch Size: {self.config.batch_size}
Dropout: {self.config.dropout}
Attention Type: {self.config.attention_type}
Use MoE: {self.config.use_moe}"""
        
        if hasattr(self.config, 'n_query_groups') and self.config.n_query_groups is not None:
            config_text += f"\nQuery Groups: {self.config.n_query_groups}"
        
        if self.config.use_moe:
            config_text += f"\nMoE Experts: {self.config.moe_num_experts}"
            config_text += f"\nMoE K: {self.config.moe_k}"
        
        config_text += f"\nVocabulary Size: {self.config.vocab_size}"
        
        self.hyperparam_text.config(state="normal")
        self.hyperparam_text.delete("1.0", tk.END)
        self.hyperparam_text.insert(tk.END, config_text)
        self.hyperparam_text.config(state="disabled")

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
