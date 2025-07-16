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
        
        # Apply VRAM optimization based on available GPU memory
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory < 20:
                    vram_profile = "low"
                elif gpu_memory < 50:
                    vram_profile = "medium"
                else:
                    vram_profile = "high"
                
                print(f"Detected {gpu_memory:.1f}GB GPU memory, applying {vram_profile} VRAM profile")
                self.config = Config.get_vram_optimized_config(vram_profile, self.config)
            else:
                print("No CUDA available, using CPU with low profile")
                self.config = Config.get_vram_optimized_config("low", self.config)
        except Exception as e:
            print(f"Error detecting GPU memory, using default config: {e}")
        
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
        self.create_advanced_tab()

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

        # Get architecture configuration from GUI
        arch_config = self.get_architecture_config()
        
        # Get parameter configuration
        param_config = self.get_parameter_config()
        if param_config is None:
            return  # Error in parameter configuration
        
        # Get sampler configuration
        sampler_config = self.get_sampler_config()
        
        # Validate configuration
        if arch_config.attention_type == 'grouped_query' and arch_config.n_head % arch_config.n_query_groups != 0:
            messagebox.showerror("Invalid Configuration", 
                               f"Number of heads ({arch_config.n_head}) must be divisible by query groups ({arch_config.n_query_groups})")
            return

        self.optimize_button.config(state="disabled")
        self.stop_optimize_button.config(state="normal")
        self.train_button.config(state="disabled")
        self.generate_button.config(state="disabled")
        
        # Update status with configuration info
        moe_info = f" with MoE ({arch_config.moe_num_experts} experts)" if arch_config.use_moe else ""
        attn_info = f" using {arch_config.attention_type} attention"
        sampler_info = f" ({sampler_config['type']} sampler)"
        self.status_text.set(f"Starting optimization: {n_trials} trials{moe_info}{attn_info}{sampler_info}")
        
        # Clear log
        self.optim_log_text.config(state="normal")
        self.optim_log_text.delete("1.0", tk.END)
        self.optim_log_text.insert(tk.END, f"=== HYPERPARAMETER OPTIMIZATION LOG ===\n")
        self.optim_log_text.insert(tk.END, f"Configuration: {n_trials} trials, {steps_per_trial} steps per trial\n")
        self.optim_log_text.insert(tk.END, f"Architecture: {arch_config.attention_type} attention{moe_info}\n")
        self.optim_log_text.insert(tk.END, f"Sampler: {sampler_config['type']}, Pruner: {sampler_config['pruner']}\n")
        
        # Add pruner parameter details
        pruner_type = sampler_config['pruner']
        if pruner_type == 'median':
            self.optim_log_text.insert(tk.END, f"  Median Pruner: startup={sampler_config.get('median_startup_trials', 5)}, warmup={sampler_config.get('median_warmup_steps', 10)}, min_trials={sampler_config.get('median_min_trials', 5)}\n")
        elif pruner_type == 'percentile':
            self.optim_log_text.insert(tk.END, f"  Percentile Pruner: percentile={sampler_config.get('percentile', 25.0)}%, startup={sampler_config.get('percentile_startup_trials', 5)}, warmup={sampler_config.get('percentile_warmup_steps', 0)}\n")
        elif pruner_type == 'successive_halving':
            self.optim_log_text.insert(tk.END, f"  Successive Halving: min_resource={sampler_config.get('halving_min_resource', 1)}, reduction={sampler_config.get('halving_reduction_factor', 4)}, min_early_stop={sampler_config.get('halving_min_early_stopping_rate', 5)}\n")
        elif pruner_type == 'none':
            self.optim_log_text.insert(tk.END, f"  No pruning enabled\n")
        
        self.optim_log_text.insert(tk.END, f"Device: {device}\n")
        
        # Show which parameters are being optimized
        params_to_optimize = []
        if param_config.get('optimize_lr', True): params_to_optimize.append("Learning Rate")
        if param_config.get('optimize_wd', True): params_to_optimize.append("Weight Decay")
        if param_config.get('optimize_embd', True): params_to_optimize.append("Embedding Dim")
        if param_config.get('optimize_layers', True): params_to_optimize.append("Layers")
        if param_config.get('optimize_heads', True): params_to_optimize.append("Heads")
        if param_config.get('optimize_dropout', True): params_to_optimize.append("Dropout")
        if param_config.get('optimize_batch', True): params_to_optimize.append("Batch Size")
        
        self.optim_log_text.insert(tk.END, f"Optimizing: {', '.join(params_to_optimize)}\n")
        self.optim_log_text.insert(tk.END, "=" * 50 + "\n\n")
        self.optim_log_text.config(state="disabled")

        self.optim_progress_bar['value'] = 0
        self.optim_progress_bar['maximum'] = n_trials

        self.optim_queue = queue.Queue()
        self.stop_optimization_event = threading.Event()
        
        thread = threading.Thread(target=self.run_optimization_thread, args=(n_trials, steps_per_trial, arch_config, param_config, sampler_config))
        thread.daemon = True
        thread.start()
        
        # Start updating the log display
        self.root.after(100, self.process_optim_queue)

    def run_optimization_thread(self, n_trials, steps_per_trial, arch_config, param_config, sampler_config):
        original_stdout = sys.stdout
        sys.stdout = QueueStream(self.optim_queue)

        try:
            new_config = run_hyperparameter_optimization(
                arch_config, n_trials, steps_per_trial, 
                param_config, self.stop_optimization_event, sampler_config
            )
            self.root.after(0, self.on_optimization_complete, new_config)
        except Exception as e:
            error_msg = str(e)
            print(f"\n--- OPTIMIZATION FAILED ---\n{error_msg}")
            self.root.after(0, lambda msg=error_msg: self.status_text.set(f"Optimization error: {msg}"))
        finally:
            sys.stdout = original_stdout
            self.root.after(0, self.optimization_finished)

    def process_optim_queue(self):
        updated = False
        trial_completed = False
        try:
            while not self.optim_queue.empty():
                line = self.optim_queue.get_nowait()
                self.optim_log_text.config(state='normal')
                self.optim_log_text.insert(tk.END, line)
                self.optim_log_text.config(state='disabled')
                self.optim_log_text.see(tk.END)
                updated = True
                
                # Update progress bar based on trial completion
                if "Trial #" in line and "completed" in line:
                    trial_completed = True
                elif "Trial #" in line and "Starting Trial" in line:
                    # Extract trial number and update progress
                    try:
                        import re
                        match = re.search(r"Trial #(\d+)", line)
                        if match:
                            current_trial = int(match.group(1))
                            self.optim_progress_bar['value'] = current_trial
                    except:
                        pass
        except:
            pass
        
        if updated:
            self.root.update_idletasks()
        
        # Update progress bar if trial completed
        if trial_completed:
            current_value = self.optim_progress_bar['value']
            if current_value < self.optim_progress_bar['maximum']:
                self.optim_progress_bar['value'] = current_value + 1
        
        # Continue processing if optimization is running
        if self.optimize_button['state'] == 'disabled':
            self.root.after(100, self.process_optim_queue)

    def stop_optimization(self):
        if hasattr(self, 'stop_optimization_event') and self.stop_optimization_event:
            print("Stop button clicked - setting stop event")
            self.stop_optimization_event.set()
            self.stop_optimize_button.config(state="disabled")
            self.status_text.set("Stopping optimization and saving best parameters...")
            
            # Add message to log
            self.optim_log_text.config(state="normal")
            self.optim_log_text.insert(tk.END, "\n=== STOP REQUESTED BY USER ===\n")
            self.optim_log_text.insert(tk.END, "Stopping optimization and saving best parameters found so far...\n")
            self.optim_log_text.config(state="disabled")
            self.optim_log_text.see(tk.END)

    def optimization_finished(self):
        self.optimize_button.config(state="normal")
        self.stop_optimize_button.config(state="disabled")
        self.train_button.config(state="normal")
        self.generate_button.config(state="normal")
        
        if hasattr(self, 'stop_optimization_event') and self.stop_optimization_event and self.stop_optimization_event.is_set():
            self.status_text.set("Optimization stopped by user. Best parameters saved.")
        else:
            self.status_text.set("Optimization completed. Best parameters saved.")

    def on_optimization_complete(self, new_config):
        if hasattr(self, 'stop_optimization_event') and self.stop_optimization_event and self.stop_optimization_event.is_set():
            messagebox.showinfo("Optimization Stopped", "Hyperparameter optimization was stopped by user. The model has been updated with the best parameters found so far.")
        else:
            messagebox.showinfo("Optimization Complete", "Hyperparameter optimization finished. The model has been updated with the best parameters found.")
        
        self.config = new_config
        self.config.vocab_size = vocab_size # Also update vocab size on new config
        
        # Update GUI controls with the new configuration
        self.use_moe_var.set(new_config.use_moe)
        self.moe_experts_var.set(str(new_config.moe_num_experts))
        self.moe_k_var.set(str(new_config.moe_k))
        self.attention_type_var.set(new_config.attention_type)
        self.query_groups_var.set(str(new_config.n_query_groups))
        self.on_attention_type_change()  # Update query groups combo state
        
        # Update parameter fixed values with optimized results
        self.lr_fixed_entry.config(state="normal")
        self.lr_fixed_entry.delete(0, tk.END)
        self.lr_fixed_entry.insert(0, str(new_config.learning_rate))
        
        self.wd_fixed_entry.config(state="normal")
        self.wd_fixed_entry.delete(0, tk.END)
        self.wd_fixed_entry.insert(0, str(new_config.weight_decay))
        
        self.embd_fixed_entry.config(state="normal")
        self.embd_fixed_entry.delete(0, tk.END)
        self.embd_fixed_entry.insert(0, str(new_config.n_embd))
        
        self.layers_fixed_entry.config(state="normal")
        self.layers_fixed_entry.delete(0, tk.END)
        self.layers_fixed_entry.insert(0, str(new_config.n_layer))
        
        self.heads_fixed_entry.config(state="normal")
        self.heads_fixed_entry.delete(0, tk.END)
        self.heads_fixed_entry.insert(0, str(new_config.n_head))
        
        self.dropout_fixed_entry.config(state="normal")
        self.dropout_fixed_entry.delete(0, tk.END)
        self.dropout_fixed_entry.insert(0, str(new_config.dropout))
        
        if hasattr(new_config, 'batch_size'):
            self.batch_fixed_entry.config(state="normal")
            self.batch_fixed_entry.delete(0, tk.END)
            self.batch_fixed_entry.insert(0, str(new_config.batch_size))
        
        # Restore parameter toggle states
        self.on_param_toggle()
        
        self.trainer = Trainer(self.config, loss_callback=self.queue_loss)
        self.update_hyperparameter_display()  # Update the display with new hyperparameters
        self.status_text.set("Model updated with optimized hyperparameters.")

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

        # Prompt input section
        tk.Label(gen_frame, text="Enter your prompt:").pack(pady=5, anchor="w")
        self.prompt_input = scrolledtext.ScrolledText(gen_frame, height=6, width=80, wrap=tk.WORD)
        self.prompt_input.pack(pady=5, fill="x", expand=False)

        # Advanced generation settings frame
        settings_frame = tk.LabelFrame(gen_frame, text="Generation Settings", padx=10, pady=5)
        settings_frame.pack(fill='x', pady=5)
        
        # First row of settings
        settings_row1 = tk.Frame(settings_frame)
        settings_row1.pack(fill='x', pady=2)
        
        tk.Label(settings_row1, text="Max Tokens:").pack(side='left', padx=5)
        self.max_tokens_entry = tk.Entry(settings_row1, width=8)
        self.max_tokens_entry.insert(0, "200")
        self.max_tokens_entry.pack(side='left', padx=5)
        
        tk.Label(settings_row1, text="Temperature:").pack(side='left', padx=(20,5))
        self.temperature_entry = tk.Entry(settings_row1, width=8)
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.pack(side='left', padx=5)
        
        tk.Label(settings_row1, text="Top-K:").pack(side='left', padx=(20,5))
        self.top_k_entry = tk.Entry(settings_row1, width=8)
        self.top_k_entry.insert(0, "50")
        self.top_k_entry.pack(side='left', padx=5)
        
        # Second row of settings
        settings_row2 = tk.Frame(settings_frame)
        settings_row2.pack(fill='x', pady=2)
        
        tk.Label(settings_row2, text="Top-P:").pack(side='left', padx=5)
        self.top_p_entry = tk.Entry(settings_row2, width=8)
        self.top_p_entry.insert(0, "0.9")
        self.top_p_entry.pack(side='left', padx=5)
        
        self.use_top_k_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_row2, text="Use Top-K", variable=self.use_top_k_var).pack(side='left', padx=20)
        
        self.use_top_p_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_row2, text="Use Top-P", variable=self.use_top_p_var).pack(side='left', padx=10)

        # Advanced generation modes
        modes_frame = tk.LabelFrame(gen_frame, text="Generation Mode", padx=10, pady=5)
        modes_frame.pack(fill='x', pady=5)
        
        self.generation_mode = tk.StringVar(value="standard")
        tk.Radiobutton(modes_frame, text="Standard", variable=self.generation_mode, value="standard").pack(side='left', padx=5)
        tk.Radiobutton(modes_frame, text="Beam Search", variable=self.generation_mode, value="beam").pack(side='left', padx=5)
        
        tk.Label(modes_frame, text="Beam Size:").pack(side='left', padx=(20,5))
        self.beam_size_entry = tk.Entry(modes_frame, width=6)
        self.beam_size_entry.insert(0, "4")
        self.beam_size_entry.pack(side='left', padx=5)

        # Generation button
        self.generate_button = tk.Button(gen_frame, text="Generate Text", command=self.start_generation_thread)
        self.generate_button.pack(pady=10)

        # Output section
        tk.Label(gen_frame, text="Generated Text:").pack(pady=5, anchor="w")
        self.output_text = scrolledtext.ScrolledText(gen_frame, height=15, width=80, wrap=tk.WORD, state="disabled")
        self.output_text.pack(pady=5, fill="both", expand=True)

    def create_training_tab(self):
        tab = self.training_tab
        
        # Title and version info
        title_frame = tk.Frame(tab)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(title_frame, text=f"DGS-GPT v{__version__}", font=("Arial", 16, "bold"))
        title_label.pack(side='left')
        
        version_info = get_version_info()
        info_label = tk.Label(title_frame, text=f"Build: {version_info['build_number']}", font=("Arial", 8))
        info_label.pack(side='right')
        
        # VRAM Profile Selection (NEW)
        vram_frame = tk.LabelFrame(tab, text="VRAM Optimization", padx=10, pady=5)
        vram_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(vram_frame, text="GPU Memory Profile:", font=("Arial", 10, "bold")).pack(anchor='w', pady=2)
        
        self.vram_profile_var = tk.StringVar(value="low")
        vram_options = [
            ("Low (15GB) - Conservative settings", "low"),
            ("Medium (40GB) - Standard settings", "medium"),
            ("High (100GB+) - Expanded settings", "high")
        ]
        
        for text, value in vram_options:
            tk.Radiobutton(vram_frame, text=text, variable=self.vram_profile_var, 
                          value=value, command=self.on_vram_profile_change).pack(anchor='w', padx=20)
        
        # VRAM info display
        self.vram_info_label = tk.Label(vram_frame, text="", font=("Arial", 8), fg="blue")
        self.vram_info_label.pack(anchor='w', padx=20, pady=2)
        
        # Initialize VRAM profile info
        self.on_vram_profile_change()

        # Controls frame with evaluation
        controls_frame = tk.Frame(tab)
        controls_frame.pack(fill='x', pady=5)
        
        # First row - main controls
        controls_row1 = tk.Frame(controls_frame)
        controls_row1.pack(fill='x', pady=2)
        
        tk.Label(controls_row1, text="Iterations:").pack(side='left', padx=5)
        self.iters_input = tk.Entry(controls_row1, width=10)
        self.iters_input.insert(0, "1000")
        self.iters_input.pack(side='left', padx=5)

        tk.Label(controls_row1, text="Plot Step:").pack(side='left', padx=5)
        self.plot_step_input = tk.Entry(controls_row1, width=10)
        self.plot_step_input.insert(0, "100")
        self.plot_step_input.pack(side='left', padx=5)

        self.train_button = tk.Button(controls_row1, text="Start Training", command=self.start_training_thread)
        self.train_button.pack(side='left', padx=5)

        self.stop_button = tk.Button(controls_row1, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.pack(side='left', padx=5)
        
        # Second row - evaluation controls
        controls_row2 = tk.Frame(controls_frame)
        controls_row2.pack(fill='x', pady=2)
        
        self.eval_button = tk.Button(controls_row2, text="Evaluate Model", command=self.evaluate_model)
        self.eval_button.pack(side='left', padx=5)
        
        tk.Label(controls_row2, text="Eval Iters:").pack(side='left', padx=5)
        self.eval_iters_input = tk.Entry(controls_row2, width=8)
        self.eval_iters_input.insert(0, "100")
        self.eval_iters_input.pack(side='left', padx=5)
        
        # Evaluation results
        self.eval_results_label = tk.Label(controls_row2, text="", fg="blue")
        self.eval_results_label.pack(side='left', padx=20)

        # Progress bar
        self.progress_bar = ttk.Progressbar(tab, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)

        # Current hyperparameters display
        hyperparam_display_frame = tk.LabelFrame(tab, text="Current Hyperparameters", padx=10, pady=5)
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

        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_hyperparameter_tab(self):
        hyperparam_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(hyperparam_frame, text='Hyperparameter Optimization')

        # Create scrollable frame for all controls
        canvas = tk.Canvas(hyperparam_frame)
        scrollbar = ttk.Scrollbar(hyperparam_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Basic Controls frame
        basic_frame = tk.LabelFrame(scrollable_frame, text="Basic Settings", padx=10, pady=5)
        basic_frame.pack(fill='x', pady=5)
        
        basic_controls = tk.Frame(basic_frame)
        basic_controls.pack(fill='x', pady=5)
        
        tk.Label(basic_controls, text="Number of Trials:").pack(side='left', padx=5)
        self.trials_input = tk.Entry(basic_controls, width=10)
        self.trials_input.insert(0, "10")
        self.trials_input.pack(side='left', padx=5)

        tk.Label(basic_controls, text="Steps per Trial:").pack(side='left', padx=5)
        self.steps_per_trial_input = tk.Entry(basic_controls, width=10)
        self.steps_per_trial_input.insert(0, "100")
        self.steps_per_trial_input.pack(side='left', padx=5)

        # Parameter Selection frame
        param_frame = tk.LabelFrame(scrollable_frame, text="Parameters to Optimize", padx=10, pady=5)
        param_frame.pack(fill='x', pady=5)

        # Learning Rate
        lr_frame = tk.Frame(param_frame)
        lr_frame.pack(fill='x', pady=2)
        self.optimize_lr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lr_frame, text="Learning Rate", variable=self.optimize_lr_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(lr_frame, text="Min:").pack(side='left', padx=(20,2))
        self.lr_min_entry = tk.Entry(lr_frame, width=10)
        self.lr_min_entry.insert(0, "1e-5")
        self.lr_min_entry.pack(side='left', padx=2)
        tk.Label(lr_frame, text="Max:").pack(side='left', padx=2)
        self.lr_max_entry = tk.Entry(lr_frame, width=10)
        self.lr_max_entry.insert(0, "1e-2")
        self.lr_max_entry.pack(side='left', padx=2)
        tk.Label(lr_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.lr_fixed_entry = tk.Entry(lr_frame, width=10)
        self.lr_fixed_entry.insert(0, str(self.config.learning_rate))
        self.lr_fixed_entry.pack(side='left', padx=2)

        # Weight Decay
        wd_frame = tk.Frame(param_frame)
        wd_frame.pack(fill='x', pady=2)
        self.optimize_wd_var = tk.BooleanVar(value=True)
        tk.Checkbutton(wd_frame, text="Weight Decay", variable=self.optimize_wd_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(wd_frame, text="Min:").pack(side='left', padx=(20,2))
        self.wd_min_entry = tk.Entry(wd_frame, width=10)
        self.wd_min_entry.insert(0, "1e-6")
        self.wd_min_entry.pack(side='left', padx=2)
        tk.Label(wd_frame, text="Max:").pack(side='left', padx=2)
        self.wd_max_entry = tk.Entry(wd_frame, width=10)
        self.wd_max_entry.insert(0, "1e-2")
        self.wd_max_entry.pack(side='left', padx=2)
        tk.Label(wd_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.wd_fixed_entry = tk.Entry(wd_frame, width=10)
        self.wd_fixed_entry.insert(0, str(self.config.weight_decay))
        self.wd_fixed_entry.pack(side='left', padx=2)

        # Embedding Dimension
        embd_frame = tk.Frame(param_frame)
        embd_frame.pack(fill='x', pady=2)
        self.optimize_embd_var = tk.BooleanVar(value=True)
        tk.Checkbutton(embd_frame, text="Embedding Dim", variable=self.optimize_embd_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(embd_frame, text="Choices:").pack(side='left', padx=(20,2))
        self.embd_choices_entry = tk.Entry(embd_frame, width=15)
        self.embd_choices_entry.insert(0, "256,512,1024")
        self.embd_choices_entry.pack(side='left', padx=2)
        tk.Label(embd_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.embd_fixed_entry = tk.Entry(embd_frame, width=10)
        self.embd_fixed_entry.insert(0, str(self.config.n_embd))
        self.embd_fixed_entry.pack(side='left', padx=2)

        # Number of Layers
        layers_frame = tk.Frame(param_frame)
        layers_frame.pack(fill='x', pady=2)
        self.optimize_layers_var = tk.BooleanVar(value=True)
        tk.Checkbutton(layers_frame, text="Layers", variable=self.optimize_layers_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(layers_frame, text="Min:").pack(side='left', padx=(20,2))
        self.layers_min_entry = tk.Entry(layers_frame, width=10)
        self.layers_min_entry.insert(0, "4")
        self.layers_min_entry.pack(side='left', padx=2)
        tk.Label(layers_frame, text="Max:").pack(side='left', padx=2)
        self.layers_max_entry = tk.Entry(layers_frame, width=10)
        self.layers_max_entry.insert(0, "16")
        self.layers_max_entry.pack(side='left', padx=2)
        tk.Label(layers_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.layers_fixed_entry = tk.Entry(layers_frame, width=10)
        self.layers_fixed_entry.insert(0, str(self.config.n_layer))
        self.layers_fixed_entry.pack(side='left', padx=2)

        # Number of Heads
        heads_frame = tk.Frame(param_frame)
        heads_frame.pack(fill='x', pady=2)
        self.optimize_heads_var = tk.BooleanVar(value=True)
        tk.Checkbutton(heads_frame, text="Heads", variable=self.optimize_heads_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(heads_frame, text="Choices:").pack(side='left', padx=(20,2))
        self.heads_choices_entry = tk.Entry(heads_frame, width=15)
        self.heads_choices_entry.insert(0, "4,8,16")
        self.heads_choices_entry.pack(side='left', padx=2)
        tk.Label(heads_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.heads_fixed_entry = tk.Entry(heads_frame, width=10)
        self.heads_fixed_entry.insert(0, str(self.config.n_head))
        self.heads_fixed_entry.pack(side='left', padx=2)

        # Dropout
        dropout_frame = tk.Frame(param_frame)
        dropout_frame.pack(fill='x', pady=2)
        self.optimize_dropout_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dropout_frame, text="Dropout", variable=self.optimize_dropout_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(dropout_frame, text="Min:").pack(side='left', padx=(20,2))
        self.dropout_min_entry = tk.Entry(dropout_frame, width=10)
        self.dropout_min_entry.insert(0, "0.0")
        self.dropout_min_entry.pack(side='left', padx=2)
        tk.Label(dropout_frame, text="Max:").pack(side='left', padx=2)
        self.dropout_max_entry = tk.Entry(dropout_frame, width=10)
        self.dropout_max_entry.insert(0, "0.5")
        self.dropout_max_entry.pack(side='left', padx=2)
        tk.Label(dropout_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.dropout_fixed_entry = tk.Entry(dropout_frame, width=10)
        self.dropout_fixed_entry.insert(0, str(self.config.dropout))
        self.dropout_fixed_entry.pack(side='left', padx=2)

        # Batch Size
        batch_frame = tk.Frame(param_frame)
        batch_frame.pack(fill='x', pady=2)
        self.optimize_batch_var = tk.BooleanVar(value=False)
        tk.Checkbutton(batch_frame, text="Batch Size", variable=self.optimize_batch_var, 
                      command=self.on_param_toggle).pack(side='left', padx=5)
        tk.Label(batch_frame, text="Choices:").pack(side='left', padx=(20,2))
        self.batch_choices_entry = tk.Entry(batch_frame, width=15)
        self.batch_choices_entry.insert(0, "8,16,24,32")
        self.batch_choices_entry.pack(side='left', padx=2)
        tk.Label(batch_frame, text="Fixed:").pack(side='left', padx=(10,2))
        self.batch_fixed_entry = tk.Entry(batch_frame, width=10)
        self.batch_fixed_entry.insert(0, str(self.config.batch_size))
        self.batch_fixed_entry.pack(side='left', padx=2)

        # Architecture settings frame
        arch_frame = tk.LabelFrame(scrollable_frame, text="Architecture Settings", padx=10, pady=5)
        arch_frame.pack(fill='x', pady=5)

        # MoE settings
        moe_frame = tk.Frame(arch_frame)
        moe_frame.pack(fill='x', pady=2)
        
        self.use_moe_var = tk.BooleanVar(value=self.config.use_moe)
        tk.Checkbutton(moe_frame, text="Use Mixture of Experts (MoE)", variable=self.use_moe_var, 
                      command=self.on_moe_toggle).pack(side='left', padx=5)
        
        tk.Label(moe_frame, text="Experts:").pack(side='left', padx=(20,5))
        self.moe_experts_var = tk.StringVar(value=str(self.config.moe_num_experts))
        moe_experts_combo = ttk.Combobox(moe_frame, textvariable=self.moe_experts_var, 
                                        values=["2", "4", "8"], width=5, state="readonly")
        moe_experts_combo.pack(side='left', padx=5)
        
        tk.Label(moe_frame, text="Top-K:").pack(side='left', padx=(10,5))
        self.moe_k_var = tk.StringVar(value=str(self.config.moe_k))
        moe_k_combo = ttk.Combobox(moe_frame, textvariable=self.moe_k_var, 
                                  values=["1", "2"], width=5, state="readonly")
        moe_k_combo.pack(side='left', padx=5)

        # Attention type settings
        attn_frame = tk.Frame(arch_frame)
        attn_frame.pack(fill='x', pady=2)
        
        tk.Label(attn_frame, text="Attention Type:").pack(side='left', padx=5)
        self.attention_type_var = tk.StringVar(value=self.config.attention_type)
        attn_combo = ttk.Combobox(attn_frame, textvariable=self.attention_type_var, 
                                 values=["multihead", "grouped_query"], width=15, state="readonly")
        attn_combo.pack(side='left', padx=5)
        attn_combo.bind('<<ComboboxSelected>>', self.on_attention_type_change)
        
        tk.Label(attn_frame, text="Query Groups:").pack(side='left', padx=(20,5))
        self.query_groups_var = tk.StringVar(value=str(self.config.n_query_groups))
        self.query_groups_combo = ttk.Combobox(attn_frame, textvariable=self.query_groups_var, 
                                              values=["1", "2", "4", "8"], width=5, state="readonly")
        self.query_groups_combo.pack(side='left', padx=5)
        
        # Update query groups based on initial attention type
        self.on_attention_type_change()

        # Sampler Settings frame
        sampler_frame = tk.LabelFrame(scrollable_frame, text="Sampler & Pruner Settings", padx=10, pady=5)
        sampler_frame.pack(fill='x', pady=5)

        sampler_controls = tk.Frame(sampler_frame)
        sampler_controls.pack(fill='x', pady=2)
        
        tk.Label(sampler_controls, text="Sampler:").pack(side='left', padx=5)
        self.sampler_var = tk.StringVar(value="tpe")
        sampler_combo = ttk.Combobox(sampler_controls, textvariable=self.sampler_var, 
                                    values=["tpe", "random", "grid", "cmaes"], width=10, state="readonly")
        sampler_combo.pack(side='left', padx=5)
        
        tk.Label(sampler_controls, text="Pruner:").pack(side='left', padx=(20,5))
        self.pruner_var = tk.StringVar(value="median")
        pruner_combo = ttk.Combobox(sampler_controls, textvariable=self.pruner_var, 
                                   values=["median", "percentile", "successive_halving", "none"], width=15, state="readonly")
        pruner_combo.pack(side='left', padx=5)

        sampler_params = tk.Frame(sampler_frame)
        sampler_params.pack(fill='x', pady=2)
        
        tk.Label(sampler_params, text="Startup Trials:").pack(side='left', padx=5)
        self.startup_trials_entry = tk.Entry(sampler_params, width=8)
        self.startup_trials_entry.insert(0, "10")
        self.startup_trials_entry.pack(side='left', padx=5)
        
        tk.Label(sampler_params, text="Seed:").pack(side='left', padx=(20,5))
        self.seed_entry = tk.Entry(sampler_params, width=10)
        self.seed_entry.insert(0, "")
        self.seed_entry.pack(side='left', padx=5)
        
        self.multivariate_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sampler_params, text="Multivariate", variable=self.multivariate_var).pack(side='left', padx=20)

        # Pruner parameters frame
        pruner_params = tk.Frame(sampler_frame)
        pruner_params.pack(fill='x', pady=2)
        
        # Add help text for pruner parameters
        help_label = tk.Label(pruner_params, text="Pruner Parameters:", font=("Arial", 9, "bold"))
        help_label.pack(anchor='w', padx=5, pady=(5,2))
        
        # Median Pruner parameters
        median_frame = tk.Frame(pruner_params)
        median_frame.pack(fill='x', pady=2)
        tk.Label(median_frame, text="Median Pruner:", fg="blue").pack(side='left', padx=5)
        
        tk.Label(median_frame, text="Startup Trials:").pack(side='left', padx=(10,2))
        self.median_startup_entry = tk.Entry(median_frame, width=6)
        self.median_startup_entry.insert(0, "5")
        self.median_startup_entry.pack(side='left', padx=2)
        
        tk.Label(median_frame, text="Warmup Steps:").pack(side='left', padx=(10,2))
        self.median_warmup_entry = tk.Entry(median_frame, width=6)
        self.median_warmup_entry.insert(0, "10")
        self.median_warmup_entry.pack(side='left', padx=2)
        
        tk.Label(median_frame, text="Min Trials:").pack(side='left', padx=(10,2))
        self.median_min_trials_entry = tk.Entry(median_frame, width=6)
        self.median_min_trials_entry.insert(0, "5")
        self.median_min_trials_entry.pack(side='left', padx=2)
        
        # Add tooltip info
        median_info = tk.Label(median_frame, text="ℹ", fg="gray", cursor="hand2")
        median_info.pack(side='left', padx=5)
        
        # Percentile Pruner parameters
        percentile_frame = tk.Frame(pruner_params)
        percentile_frame.pack(fill='x', pady=2)
        tk.Label(percentile_frame, text="Percentile Pruner:", fg="green").pack(side='left', padx=5)
        
        tk.Label(percentile_frame, text="Percentile:").pack(side='left', padx=(10,2))
        self.percentile_entry = tk.Entry(percentile_frame, width=8)
        self.percentile_entry.insert(0, "25.0")
        self.percentile_entry.pack(side='left', padx=2)
        
        tk.Label(percentile_frame, text="Startup Trials:").pack(side='left', padx=(10,2))
        self.percentile_startup_entry = tk.Entry(percentile_frame, width=6)
        self.percentile_startup_entry.insert(0, "5")
        self.percentile_startup_entry.pack(side='left', padx=2)
        
        tk.Label(percentile_frame, text="Warmup Steps:").pack(side='left', padx=(10,2))
        self.percentile_warmup_entry = tk.Entry(percentile_frame, width=6)
        self.percentile_warmup_entry.insert(0, "0")
        self.percentile_warmup_entry.pack(side='left', padx=2)
        
        percentile_info = tk.Label(percentile_frame, text="ℹ", fg="gray", cursor="hand2")
        percentile_info.pack(side='left', padx=5)
        
        # Successive Halving Pruner parameters
        halving_frame = tk.Frame(pruner_params)
        halving_frame.pack(fill='x', pady=2)
        tk.Label(halving_frame, text="Successive Halving:", fg="red").pack(side='left', padx=5)
        
        tk.Label(halving_frame, text="Min Resource:").pack(side='left', padx=(10,2))
        self.halving_min_resource_entry = tk.Entry(halving_frame, width=6)
        self.halving_min_resource_entry.insert(0, "1")
        self.halving_min_resource_entry.pack(side='left', padx=2)
        
        tk.Label(halving_frame, text="Reduction Factor:").pack(side='left', padx=(10,2))
        self.halving_reduction_entry = tk.Entry(halving_frame, width=6)
        self.halving_reduction_entry.insert(0, "4")
        self.halving_reduction_entry.pack(side='left', padx=2)
        
        tk.Label(halving_frame, text="Min Early Stop:").pack(side='left', padx=(10,2))
        self.halving_min_early_stop_entry = tk.Entry(halving_frame, width=6)
        self.halving_min_early_stop_entry.insert(0, "5")
        self.halving_min_early_stop_entry.pack(side='left', padx=2)
        
        halving_info = tk.Label(halving_frame, text="ℹ", fg="gray", cursor="hand2")
        halving_info.pack(side='left', padx=5)
        
        # Pruner description frame
        desc_frame = tk.Frame(pruner_params)
        desc_frame.pack(fill='x', pady=(5,0))
        pruner_desc = tk.Text(desc_frame, height=3, font=("Arial", 8), bg="#f0f0f0", wrap=tk.WORD)
        pruner_desc.pack(fill='x', padx=5)
        pruner_desc.insert("1.0", 
            "• Median: Prunes trials below median performance after warmup steps\n"
            "• Percentile: Prunes trials below specified percentile threshold\n"
            "• Successive Halving: Resource-efficient pruning with successive reductions")
        pruner_desc.config(state="disabled")

        # Controls frame
        controls_frame = tk.Frame(scrollable_frame)
        controls_frame.pack(fill='x', pady=10)
        
        self.optimize_button = tk.Button(controls_frame, text="Start Optimization", command=self.start_optimization_thread)
        self.optimize_button.pack(side='left', padx=5)

        self.stop_optimize_button = tk.Button(controls_frame, text="Stop Optimization", command=self.stop_optimization, state="disabled")
        self.stop_optimize_button.pack(side='left', padx=5)

        # Progress bar for optimization
        self.optim_progress_bar = ttk.Progressbar(scrollable_frame, orient='horizontal', mode='determinate')
        self.optim_progress_bar.pack(fill='x', pady=5)

        # Log display
        tk.Label(scrollable_frame, text="Optimization Log:").pack(pady=(10,5), anchor="w")
        self.optim_log_text = scrolledtext.ScrolledText(scrollable_frame, height=15, wrap=tk.WORD, state="disabled")
        self.optim_log_text.pack(fill="both", expand=True, pady=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Initialize parameter states
        self.on_param_toggle()

    def on_moe_toggle(self):
        """Handle MoE checkbox toggle"""
        # Enable/disable MoE-related controls based on checkbox state
        pass  # Controls are always enabled for user choice

    def on_attention_type_change(self, event=None):
        """Handle attention type change"""
        attention_type = self.attention_type_var.get()
        if attention_type == "grouped_query":
            # Enable query groups combo
            self.query_groups_combo.config(state="readonly")
        else:
            # Disable query groups combo for multihead attention
            self.query_groups_combo.config(state="disabled")
            self.query_groups_var.set("1")  # Reset to 1 for multihead

    def on_param_toggle(self):
        """Handle parameter optimization toggle"""
        # Enable/disable parameter entry fields based on checkbox states
        params = [
            (self.optimize_lr_var, [self.lr_min_entry, self.lr_max_entry], [self.lr_fixed_entry]),
            (self.optimize_wd_var, [self.wd_min_entry, self.wd_max_entry], [self.wd_fixed_entry]),
            (self.optimize_embd_var, [self.embd_choices_entry], [self.embd_fixed_entry]),
            (self.optimize_layers_var, [self.layers_min_entry, self.layers_max_entry], [self.layers_fixed_entry]),
            (self.optimize_heads_var, [self.heads_choices_entry], [self.heads_fixed_entry]),
            (self.optimize_dropout_var, [self.dropout_min_entry, self.dropout_max_entry], [self.dropout_fixed_entry]),
            (self.optimize_batch_var, [self.batch_choices_entry], [self.batch_fixed_entry])
        ]
        
        for var, optimize_entries, fixed_entries in params:
            if var.get():
                # Enable optimization entries, disable fixed
                for entry in optimize_entries:
                    entry.config(state="normal")
                for entry in fixed_entries:
                    entry.config(state="disabled")
            else:
                # Disable optimization entries, enable fixed
                for entry in optimize_entries:
                    entry.config(state="disabled")
                for entry in fixed_entries:
                    entry.config(state="normal")

    def get_parameter_config(self):
        """Get parameter configuration from GUI controls"""
        param_config = {}
        
        try:
            # Learning Rate
            param_config['optimize_lr'] = self.optimize_lr_var.get()
            if param_config['optimize_lr']:
                param_config['lr_min'] = float(self.lr_min_entry.get())
                param_config['lr_max'] = float(self.lr_max_entry.get())
            else:
                param_config['lr_value'] = float(self.lr_fixed_entry.get())
            
            # Weight Decay
            param_config['optimize_wd'] = self.optimize_wd_var.get()
            if param_config['optimize_wd']:
                param_config['wd_min'] = float(self.wd_min_entry.get())
                param_config['wd_max'] = float(self.wd_max_entry.get())
            else:
                param_config['wd_value'] = float(self.wd_fixed_entry.get())
            
            # Embedding Dimension
            param_config['optimize_embd'] = self.optimize_embd_var.get()
            if param_config['optimize_embd']:
                choices_str = self.embd_choices_entry.get()
                param_config['embd_choices'] = [int(x.strip()) for x in choices_str.split(',')]
            else:
                param_config['embd_value'] = int(self.embd_fixed_entry.get())
            
            # Layers
            param_config['optimize_layers'] = self.optimize_layers_var.get()
            if param_config['optimize_layers']:
                param_config['layer_min'] = int(self.layers_min_entry.get())
                param_config['layer_max'] = int(self.layers_max_entry.get())
            else:
                param_config['layer_value'] = int(self.layers_fixed_entry.get())
            
            # Heads
            param_config['optimize_heads'] = self.optimize_heads_var.get()
            if param_config['optimize_heads']:
                choices_str = self.heads_choices_entry.get()
                param_config['head_choices'] = [int(x.strip()) for x in choices_str.split(',')]
            else:
                param_config['head_value'] = int(self.heads_fixed_entry.get())
            
            # Dropout
            param_config['optimize_dropout'] = self.optimize_dropout_var.get()
            if param_config['optimize_dropout']:
                param_config['dropout_min'] = float(self.dropout_min_entry.get())
                param_config['dropout_max'] = float(self.dropout_max_entry.get())
            else:
                param_config['dropout_value'] = float(self.dropout_fixed_entry.get())
            
            # Batch Size
            param_config['optimize_batch'] = self.optimize_batch_var.get()
            if param_config['optimize_batch']:
                choices_str = self.batch_choices_entry.get()
                param_config['batch_choices'] = [int(x.strip()) for x in choices_str.split(',')]
            else:
                param_config['batch_value'] = int(self.batch_fixed_entry.get())
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid parameter value: {e}")
            return None
        
        return param_config

    def get_sampler_config(self):
        """Get sampler configuration from GUI controls"""
        sampler_config = {}
        
        sampler_config['type'] = self.sampler_var.get()
        sampler_config['pruner'] = self.pruner_var.get()
        
        try:
            sampler_config['n_startup_trials'] = int(self.startup_trials_entry.get())
        except:
            sampler_config['n_startup_trials'] = 10
        
        seed_text = self.seed_entry.get().strip()
        if seed_text:
            try:
                sampler_config['seed'] = int(seed_text)
            except:
                sampler_config['seed'] = None
        else:
            sampler_config['seed'] = None
        
        sampler_config['multivariate'] = self.multivariate_var.get()
        
        # Add pruner-specific parameters
        try:
            # Median Pruner parameters
            sampler_config['median_startup_trials'] = int(self.median_startup_entry.get())
            sampler_config['median_warmup_steps'] = int(self.median_warmup_entry.get())
            sampler_config['median_min_trials'] = int(self.median_min_trials_entry.get())
            
            # Percentile Pruner parameters
            sampler_config['percentile'] = float(self.percentile_entry.get())
            sampler_config['percentile_startup_trials'] = int(self.percentile_startup_entry.get())
            sampler_config['percentile_warmup_steps'] = int(self.percentile_warmup_entry.get())
            
            # Successive Halving Pruner parameters
            sampler_config['halving_min_resource'] = int(self.halving_min_resource_entry.get())
            sampler_config['halving_reduction_factor'] = int(self.halving_reduction_entry.get())
            sampler_config['halving_min_early_stopping_rate'] = int(self.halving_min_early_stop_entry.get())
        except ValueError:
            # Use defaults if invalid values entered
            sampler_config['median_startup_trials'] = 5
            sampler_config['median_warmup_steps'] = 10
            sampler_config['median_min_trials'] = 5
            sampler_config['percentile'] = 25.0
            sampler_config['percentile_startup_trials'] = 5
            sampler_config['percentile_warmup_steps'] = 0
            sampler_config['halving_min_resource'] = 1
            sampler_config['halving_reduction_factor'] = 4
            sampler_config['halving_min_early_stopping_rate'] = 5
        
        return sampler_config

    def get_architecture_config(self):
        """Get architecture configuration from GUI controls"""
        arch_config = Config()
        
        # Copy base config values
        for attr in dir(self.config):
            if not attr.startswith('_') and hasattr(arch_config, attr):
                setattr(arch_config, attr, getattr(self.config, attr))
        
        # Update with GUI values
        arch_config.use_moe = self.use_moe_var.get()
        arch_config.moe_num_experts = int(self.moe_experts_var.get())
        arch_config.moe_k = int(self.moe_k_var.get())
        arch_config.attention_type = self.attention_type_var.get()
        arch_config.n_query_groups = int(self.query_groups_var.get())
        
        return arch_config

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

Architecture Settings:
Attention Type: {self.config.attention_type}"""
        
        if hasattr(self.config, 'n_query_groups') and self.config.n_query_groups is not None:
            config_text += f"\nQuery Groups: {self.config.n_query_groups}"
        
        config_text += f"\nUse MoE: {self.config.use_moe}"
        if self.config.use_moe:
            config_text += f"\nMoE Experts: {self.config.moe_num_experts}"
            config_text += f"\nMoE Top-K: {self.config.moe_k}"
        
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

        # Get generation settings
        try:
            max_tokens = int(self.max_tokens_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get()) if self.use_top_k_var.get() else None
            top_p = float(self.top_p_entry.get()) if self.use_top_p_var.get() else None
            beam_size = int(self.beam_size_entry.get()) if self.generation_mode.get() == "beam" else 1
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid generation setting value: {e}")
            return

        try:
            with torch.no_grad():
                self.trainer.model.eval()
                idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                
                if self.generation_mode.get() == "beam":
                    # Beam search generation
                    gen_ids = self.trainer.model.beam_search_generate(
                        idx, 
                        max_new_tokens=max_tokens, 
                        beam_size=beam_size,
                        temperature=temperature
                    )
                else:
                    # Standard generation
                    gen_ids = self.trainer.model.generate(
                        idx, 
                        max_new_tokens=max_tokens, 
                        temperature=temperature, 
                        top_k=top_k, 
                        top_p=top_p
                    )
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

    def evaluate_model(self):
        """Evaluate the current model and display results"""
        try:
            eval_iters = int(self.eval_iters_input.get())
            if eval_iters <= 0:
                raise ValueError("Evaluation iterations must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid evaluation iterations: {e}")
            return
        
        self.eval_button.config(state="disabled")
        self.status_text.set("Evaluating model...")
        
        # Run evaluation in thread to avoid blocking GUI
        def eval_thread():
            try:
                avg_loss, perplexity = self.trainer.evaluate(eval_iters)
                self.root.after(0, self.on_evaluation_complete, avg_loss, perplexity)
            except Exception as e:
                self.root.after(0, lambda: self.status_text.set(f"Evaluation error: {e}"))
                self.root.after(0, lambda: self.eval_button.config(state="normal"))
        
        thread = threading.Thread(target=eval_thread)
        thread.daemon = True
        thread.start()
    
    def on_evaluation_complete(self, avg_loss, perplexity):
        """Handle evaluation completion"""
        self.eval_results_label.config(text=f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        self.status_text.set(f"Evaluation complete. Perplexity: {perplexity:.2f}")
        self.eval_button.config(state="normal")

    def on_vram_profile_change(self):
        """Handle VRAM profile selection changes"""
        profile = self.vram_profile_var.get()
        
        if profile == "low":
            info_text = "15GB VRAM: batch_size=2, context=512, layers=4, embedding=1024"
        elif profile == "medium":
            info_text = "40GB VRAM: batch_size=16, context=8192, layers=32, embedding=4096"
        elif profile == "high":
            info_text = "100GB+ VRAM: batch_size=32, context=16384, layers=40, embedding=5120"
        else:
            info_text = ""
            
        self.vram_info_label.config(text=info_text)
        
        # Update config if trainer exists
        if hasattr(self, 'trainer') and self.trainer:
            try:
                # Apply VRAM optimized config
                new_config = Config.get_vram_optimized_config(profile, self.config)
                self.config = new_config
                
                # Recreate trainer with new config
                self.trainer = Trainer(self.config, self.queue_loss)
                
                # Update displays
                self.update_hyperparameter_display()
                
                print(f"Applied {profile.upper()} VRAM profile")
            except Exception as e:
                print(f"Error applying VRAM profile: {e}")

    def create_advanced_tab(self):
        """Create advanced settings tab with notebook-inspired features"""
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Advanced")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Advanced Architecture Settings
        arch_frame = tk.LabelFrame(scrollable_frame, text="Advanced Architecture", padx=10, pady=5)
        arch_frame.pack(fill='x', pady=5)
        
        # RoPE Settings
        rope_frame = tk.Frame(arch_frame)
        rope_frame.pack(fill='x', pady=2)
        
        self.use_rope_var = tk.BooleanVar(value=True)
        tk.Checkbutton(rope_frame, text="Use Rotary Positional Embedding (RoPE)", 
                      variable=self.use_rope_var).pack(side='left', padx=5)
        
        # Sliding Window Attention
        sliding_frame = tk.Frame(arch_frame)
        sliding_frame.pack(fill='x', pady=2)
        
        self.use_sliding_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sliding_frame, text="Use Sliding Window Attention", 
                      variable=self.use_sliding_var).pack(side='left', padx=5)
        
        tk.Label(sliding_frame, text="Window Size:").pack(side='left', padx=(20,5))
        self.sliding_window_entry = tk.Entry(sliding_frame, width=8)
        self.sliding_window_entry.insert(0, "256")
        self.sliding_window_entry.pack(side='left', padx=5)
        
        # Sparse MoE Settings
        sparse_moe_frame = tk.Frame(arch_frame)
        sparse_moe_frame.pack(fill='x', pady=2)
        
        self.use_sparse_moe_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sparse_moe_frame, text="Use Sparse Mixture of Experts", 
                      variable=self.use_sparse_moe_var).pack(side='left', padx=5)
        
        tk.Label(sparse_moe_frame, text="Capacity Factor:").pack(side='left', padx=(20,5))
        self.capacity_factor_entry = tk.Entry(sparse_moe_frame, width=8)
        self.capacity_factor_entry.insert(0, "1.2")
        self.capacity_factor_entry.pack(side='left', padx=5)
        
        # Training Optimizations
        training_frame = tk.LabelFrame(scrollable_frame, text="Training Optimizations", padx=10, pady=5)
        training_frame.pack(fill='x', pady=5)
        
        # Gradient Checkpointing
        checkpoint_frame = tk.Frame(training_frame)
        checkpoint_frame.pack(fill='x', pady=2)
        
        self.use_checkpointing_var = tk.BooleanVar(value=True)
        tk.Checkbutton(checkpoint_frame, text="Enable Gradient Checkpointing (VRAM optimization)", 
                      variable=self.use_checkpointing_var).pack(side='left', padx=5)
        
        # Enhanced Scheduler Settings
        scheduler_frame = tk.Frame(training_frame)
        scheduler_frame.pack(fill='x', pady=2)
        
        self.use_cosine_scheduler_var = tk.BooleanVar(value=True)
        tk.Checkbutton(scheduler_frame, text="Use Cosine Warmup Scheduler", 
                      variable=self.use_cosine_scheduler_var).pack(side='left', padx=5)
        
        # Warmup settings
        warmup_frame = tk.Frame(training_frame)
        warmup_frame.pack(fill='x', pady=2)
        
        tk.Label(warmup_frame, text="Warmup Steps:").pack(side='left', padx=5)
        self.warmup_steps_entry = tk.Entry(warmup_frame, width=8)
        self.warmup_steps_entry.insert(0, "1000")
        self.warmup_steps_entry.pack(side='left', padx=5)
        
        tk.Label(warmup_frame, text="Min LR Ratio:").pack(side='left', padx=(20,5))
        self.min_lr_ratio_entry = tk.Entry(warmup_frame, width=8)
        self.min_lr_ratio_entry.insert(0, "0.1")
        self.min_lr_ratio_entry.pack(side='left', padx=5)
        
        # Enhanced Checkpointing
        checkpoint_settings_frame = tk.LabelFrame(scrollable_frame, text="Enhanced Checkpointing", padx=10, pady=5)
        checkpoint_settings_frame.pack(fill='x', pady=5)
        
        # Auto-save settings
        autosave_frame = tk.Frame(checkpoint_settings_frame)
        autosave_frame.pack(fill='x', pady=2)
        
        self.auto_save_var = tk.BooleanVar(value=True)
        tk.Checkbutton(autosave_frame, text="Auto-save with metadata", 
                      variable=self.auto_save_var).pack(side='left', padx=5)
        
        tk.Label(autosave_frame, text="Save every:").pack(side='left', padx=(20,5))
        self.save_interval_entry = tk.Entry(autosave_frame, width=8)
        self.save_interval_entry.insert(0, "1000")
        self.save_interval_entry.pack(side='left', padx=5)
        tk.Label(autosave_frame, text="iterations").pack(side='left', padx=2)
        
        # Perplexity Evaluation
        eval_frame = tk.LabelFrame(scrollable_frame, text="Model Evaluation", padx=10, pady=5)
        eval_frame.pack(fill='x', pady=5)
        
        # Auto evaluation
        auto_eval_frame = tk.Frame(eval_frame)
        auto_eval_frame.pack(fill='x', pady=2)
        
        self.auto_eval_var = tk.BooleanVar(value=True)
        tk.Checkbutton(auto_eval_frame, text="Auto-evaluate perplexity", 
                      variable=self.auto_eval_var).pack(side='left', padx=5)
        
        tk.Label(auto_eval_frame, text="Eval every:").pack(side='left', padx=(20,5))
        self.eval_interval_entry = tk.Entry(auto_eval_frame, width=8)
        self.eval_interval_entry.insert(0, "500")
        self.eval_interval_entry.pack(side='left', padx=5)
        tk.Label(auto_eval_frame, text="iterations").pack(side='left', padx=2)
        
        # Manual evaluation
        manual_eval_frame = tk.Frame(eval_frame)
        manual_eval_frame.pack(fill='x', pady=2)
        
        tk.Button(manual_eval_frame, text="Calculate Perplexity", 
                 command=self.calculate_perplexity_manual).pack(side='left', padx=5)
        
        self.perplexity_result_label = tk.Label(manual_eval_frame, text="Perplexity: --", 
                                               font=("Arial", 10))
        self.perplexity_result_label.pack(side='left', padx=20)
        
        # Apply settings button
        apply_frame = tk.Frame(scrollable_frame)
        apply_frame.pack(fill='x', pady=10)
        
        tk.Button(apply_frame, text="Apply Advanced Settings", 
                 command=self.apply_advanced_settings, bg="lightgreen").pack(pady=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def apply_advanced_settings(self):
        """Apply advanced settings to the model configuration"""
        try:
            # Update config with advanced settings
            self.config.use_gradient_checkpointing = self.use_checkpointing_var.get()
            self.config.warmup_iters = int(self.warmup_steps_entry.get())
            
            # Create new trainer with updated config
            self.trainer = Trainer(self.config, self.queue_loss)
            
            # Update displays
            self.update_hyperparameter_display()
            
            messagebox.showinfo("Success", "Advanced settings applied successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply advanced settings: {e}")

    def calculate_perplexity_manual(self):
        """Manually calculate and display perplexity"""
        if not hasattr(self, 'trainer') or not self.trainer:
            messagebox.showwarning("Warning", "Please initialize the trainer first")
            return
            
        try:
            self.perplexity_result_label.config(text="Calculating...")
            self.root.update()
            
            avg_loss, perplexity = self.trainer.evaluate(eval_iters=100)
            self.perplexity_result_label.config(text=f"Perplexity: {perplexity:.2f}")
            
        except Exception as e:
            self.perplexity_result_label.config(text="Error calculating")
            messagebox.showerror("Error", f"Failed to calculate perplexity: {e}")

    # ...existing methods...
