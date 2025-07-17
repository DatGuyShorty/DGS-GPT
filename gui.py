import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog, simpledialog
import threading
import torch
import os
import queue
import sys
import re
import math
import datetime
import json

# Matplotlib for plotting
import matplotlib
matplotlib.use('TkAgg')  # Ensure TkAgg backend for better tkinter integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# Set a clean plotting style with anti-aliasing (with fallback)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style if seaborn styles not available

# Configure matplotlib for smoother plots
matplotlib.rcParams['lines.antialiased'] = True
matplotlib.rcParams['text.antialiased'] = True
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['patch.antialiased'] = True

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
        vocab_size,
        get_training_speed_config,
        optimize_for_training_speed,
        clear_cache_and_optimize
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

        # Initialize settings file path
        self.settings_file = "gui_settings.json"
        
        # Prevent VRAM profile changes during initialization
        self._initializing = True
        self._updating_vram_profile = False
        self._last_working_profile = "low"
        
        self.loss_queue = queue.Queue()
        self.losses = []
        self.perplexities = []
        self.iterations = []
        self.learning_rates = []
        self.grad_norms = []
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
        
        self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)

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
        self.create_model_settings_tab()  # Combined VRAM and hyperparameters
        self.create_advanced_tab()

        # Load persistent settings after creating all GUI components
        self.load_gui_settings()

        # Mark initialization as complete - VRAM profile changes now allowed
        self._initializing = False
        
        # Store the current working profile
        if hasattr(self, 'vram_profile_var'):
            self._last_working_profile = self.vram_profile_var.get()

        self.root.after(100, self.process_loss_queue)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Save Hyperparameters", command=self.save_hyperparameters)
        file_menu.add_command(label="Load Hyperparameters", command=self.load_hyperparameters)
        file_menu.add_command(label="Apply Current Hyperparameters", command=self.apply_hyperparameters)
        file_menu.add_separator()
        file_menu.add_command(label="Export Settings", command=self.export_settings)
        file_menu.add_command(label="Import Settings", command=self.import_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def save_hyperparameters(self):
        """Save current hyperparameters to a JSON file"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Hyperparameters As"
        )
        if filepath:
            try:
                hyperparams = {
                    'learning_rate': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay,
                    'n_embd': self.config.n_embd,
                    'n_layer': self.config.n_layer,
                    'n_head': self.config.n_head,
                    'block_size': self.config.block_size,
                    'batch_size': self.config.batch_size,
                    'dropout': self.config.dropout,
                    'attention_type': self.config.attention_type,
                    'n_query_groups': self.config.n_query_groups,
                    'use_moe': self.config.use_moe,
                    'moe_num_experts': self.config.moe_num_experts,
                    'moe_k': self.config.moe_k,
                    'vram_profile': getattr(self.config, 'vram_profile', 'low'),
                    'use_gradient_checkpointing': getattr(self.config, 'use_gradient_checkpointing', True),
                    'warmup_iters': self.config.warmup_iters,
                    'grad_clip': self.config.grad_clip,
                    'max_iters': self.config.max_iters
                }
                
                with open(filepath, 'w') as f:
                    json.dump(hyperparams, f, indent=4)
                
                messagebox.showinfo("Success", f"Hyperparameters saved to {os.path.basename(filepath)}")
                self.status_text.set(f"Hyperparameters saved to {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save hyperparameters:\n{e}")

    def load_hyperparameters(self):
        """Load hyperparameters from a JSON file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Hyperparameters"
        )
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    hyperparams = json.load(f)
                
                # Update config with loaded hyperparameters
                for key, value in hyperparams.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Update GUI controls with loaded values
                self.update_gui_from_config()
                
                messagebox.showinfo("Success", f"Hyperparameters loaded from {os.path.basename(filepath)}")
                self.status_text.set(f"Hyperparameters loaded from {os.path.basename(filepath)}")
                
                # Ask if user wants to apply the loaded hyperparameters
                if messagebox.askyesno("Apply Hyperparameters", "Would you like to apply the loaded hyperparameters to the model?"):
                    self.apply_hyperparameters()
                    
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load hyperparameters:\n{e}")
        elif filepath:
            messagebox.showerror("File Not Found", f"The file '{filepath}' does not exist.")

    def apply_hyperparameters(self):
        """Apply current hyperparameters to the model by recreating trainer"""
        try:
            # Update config with current GUI values
            self.update_config_from_gui()
            
            # Recreate trainer with updated config
            old_trainer = self.trainer
            self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
            
            # Try to transfer model weights if compatible
            try:
                if hasattr(old_trainer, 'model') and hasattr(self.trainer, 'model'):
                    # Check if architectures are compatible
                    old_state = old_trainer.model.state_dict()
                    new_state = self.trainer.model.state_dict()
                    
                    compatible_keys = []
                    for key in old_state.keys():
                        if key in new_state and old_state[key].shape == new_state[key].shape:
                            compatible_keys.append(key)
                    
                    if compatible_keys:
                        # Transfer compatible weights
                        transfer_dict = {k: old_state[k] for k in compatible_keys}
                        self.trainer.model.load_state_dict(transfer_dict, strict=False)
                        print(f"Transferred {len(compatible_keys)} compatible weight tensors")
                        
            except Exception as e:
                print(f"Warning: Could not transfer model weights: {e}")
            
            # Update hyperparameter display
            self.update_hyperparameter_display()
            
            messagebox.showinfo("Success", "Hyperparameters applied successfully!\nModel has been recreated with new settings.")
            self.status_text.set("Hyperparameters applied to model")
            
        except Exception as e:
            messagebox.showerror("Apply Error", f"Failed to apply hyperparameters:\n{e}")

    def update_gui_from_config(self):
        """Update GUI controls with values from config"""
        try:
            # Update hyperparameter optimization controls (legacy)
            if hasattr(self, 'lr_fixed_entry'):
                self.lr_fixed_entry.config(state="normal")
                self.lr_fixed_entry.delete(0, tk.END)
                self.lr_fixed_entry.insert(0, str(self.config.learning_rate))
                
            if hasattr(self, 'wd_fixed_entry'):
                self.wd_fixed_entry.config(state="normal")
                self.wd_fixed_entry.delete(0, tk.END)
                self.wd_fixed_entry.insert(0, str(self.config.weight_decay))
                
            if hasattr(self, 'embd_fixed_entry'):
                self.embd_fixed_entry.config(state="normal")
                self.embd_fixed_entry.delete(0, tk.END)
                self.embd_fixed_entry.insert(0, str(self.config.n_embd))
                
            if hasattr(self, 'layers_fixed_entry'):
                self.layers_fixed_entry.config(state="normal")
                self.layers_fixed_entry.delete(0, tk.END)
                self.layers_fixed_entry.insert(0, str(self.config.n_layer))
                
            if hasattr(self, 'heads_fixed_entry'):
                self.heads_fixed_entry.config(state="normal")
                self.heads_fixed_entry.delete(0, tk.END)
                self.heads_fixed_entry.insert(0, str(self.config.n_head))
                
            if hasattr(self, 'dropout_fixed_entry'):
                self.dropout_fixed_entry.config(state="normal")
                self.dropout_fixed_entry.delete(0, tk.END)
                self.dropout_fixed_entry.insert(0, str(self.config.dropout))
                
            if hasattr(self, 'batch_fixed_entry'):
                self.batch_fixed_entry.config(state="normal")
                self.batch_fixed_entry.delete(0, tk.END)
                self.batch_fixed_entry.insert(0, str(self.config.batch_size))
            
            # Update unified model settings tab variables
            if hasattr(self, 'lr_var'):
                self.lr_var.set(str(self.config.learning_rate))
            if hasattr(self, 'wd_var'):
                self.wd_var.set(str(self.config.weight_decay))
            if hasattr(self, 'dropout_var'):
                self.dropout_var.set(str(self.config.dropout))
            if hasattr(self, 'batch_size_var'):
                self.batch_size_var.set(str(self.config.batch_size))
            if hasattr(self, 'block_size_var'):
                self.block_size_var.set(str(self.config.block_size))
            if hasattr(self, 'embd_dim_var'):
                self.embd_dim_var.set(str(self.config.n_embd))
            if hasattr(self, 'layers_var'):
                self.layers_var.set(str(self.config.n_layer))
            if hasattr(self, 'heads_var'):
                self.heads_var.set(str(self.config.n_head))
            
            # Update training tab batch size input
            if hasattr(self, 'batch_size_input'):
                self.batch_size_input.delete(0, tk.END)
                self.batch_size_input.insert(0, str(self.config.batch_size))
                
            # Log the update if console exists
            if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
                self.log_to_console(f"Batch size field updated to: {self.config.batch_size}", "DEBUG")
                self.log_to_console(f"Other config values - layers: {self.config.n_layer}, embedding: {self.config.n_embd}, heads: {self.config.n_head}", "DEBUG")
            
            # Update architecture controls
            if hasattr(self, 'use_moe_var'):
                self.use_moe_var.set(self.config.use_moe)
            if hasattr(self, 'moe_experts_var'):
                self.moe_experts_var.set(str(self.config.moe_num_experts))
            if hasattr(self, 'moe_k_var'):
                self.moe_k_var.set(str(self.config.moe_k))
            if hasattr(self, 'attention_type_var'):
                self.attention_type_var.set(self.config.attention_type)
            if hasattr(self, 'query_groups_var'):
                self.query_groups_var.set(str(self.config.n_query_groups))
            
            # Update VRAM profile (avoid recursive calls)
            if hasattr(self, 'vram_profile_var') and not (hasattr(self, '_updating_vram_profile') and self._updating_vram_profile):
                vram_profile = getattr(self.config, 'vram_profile', 'low')
                self.vram_profile_var.set(vram_profile)
                
        except Exception as e:
            # Handle errors gracefully - some GUI components may not exist yet
            if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
                self.log_to_console(f"Warning: Error updating GUI from config: {str(e)}", "WARNING")
            else:
                print(f"Warning: Error updating GUI from config: {str(e)}")
            if hasattr(self, 'on_param_toggle'):
                self.on_param_toggle()
                
            # Update attention type dependent controls
            if hasattr(self, 'on_attention_type_change'):
                self.on_attention_type_change()
                
        except Exception as e:
            print(f"Warning: Error updating GUI from config: {e}")

    def update_config_from_gui(self):
        """Update config with current GUI values"""
        try:
            # Architecture settings
            if hasattr(self, 'use_moe_var'):
                self.config.use_moe = self.use_moe_var.get()
            if hasattr(self, 'moe_experts_var'):
                self.config.moe_num_experts = int(self.moe_experts_var.get())
            if hasattr(self, 'moe_k_var'):
                self.config.moe_k = int(self.moe_k_var.get())
            if hasattr(self, 'attention_type_var'):
                self.config.attention_type = self.attention_type_var.get()
            if hasattr(self, 'query_groups_var'):
                self.config.n_query_groups = int(self.query_groups_var.get())
            
            # VRAM profile
            if hasattr(self, 'vram_profile_var'):
                self.config.vram_profile = self.vram_profile_var.get()
                
            # Advanced settings
            if hasattr(self, 'use_checkpointing_var'):
                self.config.use_gradient_checkpointing = self.use_checkpointing_var.get()
            if hasattr(self, 'warmup_steps_entry'):
                try:
                    self.config.warmup_iters = int(self.warmup_steps_entry.get())
                except ValueError:
                    pass
                    
        except Exception as e:
            print(f"Warning: Error updating config from GUI: {e}")

    def save_gui_settings(self):
        """Save all GUI settings to a JSON file"""
        try:
            settings = {
                'generation_settings': {
                    'max_tokens': self.max_tokens_entry.get() if hasattr(self, 'max_tokens_entry') else "200",
                    'temperature': self.temperature_entry.get() if hasattr(self, 'temperature_entry') else "1.0",
                    'top_k': self.top_k_entry.get() if hasattr(self, 'top_k_entry') else "50",
                    'top_p': self.top_p_entry.get() if hasattr(self, 'top_p_entry') else "0.9",
                    'use_top_k': self.use_top_k_var.get() if hasattr(self, 'use_top_k_var') else True,
                    'use_top_p': self.use_top_p_var.get() if hasattr(self, 'use_top_p_var') else False,
                    'generation_mode': self.generation_mode.get() if hasattr(self, 'generation_mode') else "standard",
                    'beam_size': self.beam_size_entry.get() if hasattr(self, 'beam_size_entry') else "4"
                },
                'training_settings': {
                    'iterations': self.iters_input.get() if hasattr(self, 'iters_input') else "1000",
                    'plot_step': self.plot_step_input.get() if hasattr(self, 'plot_step_input') else "100",
                    'batch_size': self.batch_size_input.get() if hasattr(self, 'batch_size_input') else str(self.config.batch_size),
                    'eval_iters': self.eval_iters_input.get() if hasattr(self, 'eval_iters_input') else "100",
                    'auto_scroll': self.auto_scroll_var.get() if hasattr(self, 'auto_scroll_var') else True,
                    'vram_profile': self.vram_profile_var.get() if hasattr(self, 'vram_profile_var') else "low"
                },
                'optimization_settings': {
                    'trials': self.trials_input.get() if hasattr(self, 'trials_input') else "10",
                    'steps_per_trial': self.steps_per_trial_input.get() if hasattr(self, 'steps_per_trial_input') else "100",
                    'optimize_lr': self.optimize_lr_var.get() if hasattr(self, 'optimize_lr_var') else True,
                    'optimize_wd': self.optimize_wd_var.get() if hasattr(self, 'optimize_wd_var') else True,
                    'optimize_embd': self.optimize_embd_var.get() if hasattr(self, 'optimize_embd_var') else True,
                    'optimize_layers': self.optimize_layers_var.get() if hasattr(self, 'optimize_layers_var') else True,
                    'optimize_heads': self.optimize_heads_var.get() if hasattr(self, 'optimize_heads_var') else True,
                    'optimize_dropout': self.optimize_dropout_var.get() if hasattr(self, 'optimize_dropout_var') else True,
                    'optimize_batch': self.optimize_batch_var.get() if hasattr(self, 'optimize_batch_var') else False,
                    'sampler': self.sampler_var.get() if hasattr(self, 'sampler_var') else "tpe",
                    'pruner': self.pruner_var.get() if hasattr(self, 'pruner_var') else "median"
                },
                'architecture_settings': {
                    'use_moe': self.use_moe_var.get() if hasattr(self, 'use_moe_var') else True,
                    'moe_experts': self.moe_experts_var.get() if hasattr(self, 'moe_experts_var') else "4",
                    'moe_k': self.moe_k_var.get() if hasattr(self, 'moe_k_var') else "1",
                    'attention_type': self.attention_type_var.get() if hasattr(self, 'attention_type_var') else "grouped_query",
                    'query_groups': self.query_groups_var.get() if hasattr(self, 'query_groups_var') else "1"
                },
                'advanced_settings': {
                    'use_checkpointing': self.use_checkpointing_var.get() if hasattr(self, 'use_checkpointing_var') else True,
                    'warmup_steps': self.warmup_steps_entry.get() if hasattr(self, 'warmup_steps_entry') else "100",
                    'auto_save': self.auto_save_var.get() if hasattr(self, 'auto_save_var') else True,
                    'auto_eval': self.auto_eval_var.get() if hasattr(self, 'auto_eval_var') else True
                }
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
                
        except Exception as e:
            print(f"Warning: Error saving GUI settings: {e}")

    def load_gui_settings(self):
        """Load GUI settings from JSON file"""
        try:
            if not os.path.exists(self.settings_file):
                return
                
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            # Load generation settings
            gen_settings = settings.get('generation_settings', {})
            if hasattr(self, 'max_tokens_entry'):
                self.max_tokens_entry.delete(0, tk.END)
                self.max_tokens_entry.insert(0, gen_settings.get('max_tokens', '200'))
            if hasattr(self, 'temperature_entry'):
                self.temperature_entry.delete(0, tk.END)
                self.temperature_entry.insert(0, gen_settings.get('temperature', '1.0'))
            if hasattr(self, 'top_k_entry'):
                self.top_k_entry.delete(0, tk.END)
                self.top_k_entry.insert(0, gen_settings.get('top_k', '50'))
            if hasattr(self, 'top_p_entry'):
                self.top_p_entry.delete(0, tk.END)
                self.top_p_entry.insert(0, gen_settings.get('top_p', '0.9'))
            if hasattr(self, 'use_top_k_var'):
                self.use_top_k_var.set(gen_settings.get('use_top_k', True))
            if hasattr(self, 'use_top_p_var'):
                self.use_top_p_var.set(gen_settings.get('use_top_p', False))
            if hasattr(self, 'generation_mode'):
                self.generation_mode.set(gen_settings.get('generation_mode', 'standard'))
            if hasattr(self, 'beam_size_entry'):
                self.beam_size_entry.delete(0, tk.END)
                self.beam_size_entry.insert(0, gen_settings.get('beam_size', '4'))
            
            # Load training settings
            train_settings = settings.get('training_settings', {})
            if hasattr(self, 'iters_input'):
                self.iters_input.delete(0, tk.END)
                self.iters_input.insert(0, train_settings.get('iterations', '1000'))
            if hasattr(self, 'plot_step_input'):
                self.plot_step_input.delete(0, tk.END)
                self.plot_step_input.insert(0, train_settings.get('plot_step', '100'))
            # Load VRAM profile first and apply it
            if hasattr(self, 'vram_profile_var'):
                saved_vram_profile = train_settings.get('vram_profile', 'low')
                self.vram_profile_var.set(saved_vram_profile)
                # Apply the VRAM profile BEFORE loading other settings so batch size gets set correctly
                self.on_vram_profile_change()
                
            # Load batch size AFTER applying VRAM profile, but only if no VRAM profile was saved
            # This prevents saved batch size from overriding VRAM profile settings
            if hasattr(self, 'batch_size_input') and 'vram_profile' not in train_settings:
                self.batch_size_input.delete(0, tk.END)
                self.batch_size_input.insert(0, train_settings.get('batch_size', str(self.config.batch_size)))
            
            # Load other training settings
            if hasattr(self, 'eval_iters_input'):
                self.eval_iters_input.delete(0, tk.END)
                self.eval_iters_input.insert(0, train_settings.get('eval_iters', '100'))
            if hasattr(self, 'auto_scroll_var'):
                self.auto_scroll_var.set(train_settings.get('auto_scroll', True))
            
            # Load optimization settings
            opt_settings = settings.get('optimization_settings', {})
            if hasattr(self, 'trials_input'):
                self.trials_input.delete(0, tk.END)
                self.trials_input.insert(0, opt_settings.get('trials', '10'))
            if hasattr(self, 'steps_per_trial_input'):
                self.steps_per_trial_input.delete(0, tk.END)
                self.steps_per_trial_input.insert(0, opt_settings.get('steps_per_trial', '100'))
            if hasattr(self, 'optimize_lr_var'):
                self.optimize_lr_var.set(opt_settings.get('optimize_lr', True))
            if hasattr(self, 'optimize_wd_var'):
                self.optimize_wd_var.set(opt_settings.get('optimize_wd', True))
            if hasattr(self, 'optimize_embd_var'):
                self.optimize_embd_var.set(opt_settings.get('optimize_embd', True))
            if hasattr(self, 'optimize_layers_var'):
                self.optimize_layers_var.set(opt_settings.get('optimize_layers', True))
            if hasattr(self, 'optimize_heads_var'):
                self.optimize_heads_var.set(opt_settings.get('optimize_heads', True))
            if hasattr(self, 'optimize_dropout_var'):
                self.optimize_dropout_var.set(opt_settings.get('optimize_dropout', True))
            if hasattr(self, 'optimize_batch_var'):
                self.optimize_batch_var.set(opt_settings.get('optimize_batch', False))
            if hasattr(self, 'sampler_var'):
                self.sampler_var.set(opt_settings.get('sampler', 'tpe'))
            if hasattr(self, 'pruner_var'):
                self.pruner_var.set(opt_settings.get('pruner', 'median'))
            
            # Load architecture settings
            arch_settings = settings.get('architecture_settings', {})
            if hasattr(self, 'use_moe_var'):
                self.use_moe_var.set(arch_settings.get('use_moe', True))
            if hasattr(self, 'moe_experts_var'):
                self.moe_experts_var.set(arch_settings.get('moe_experts', '4'))
            if hasattr(self, 'moe_k_var'):
                self.moe_k_var.set(arch_settings.get('moe_k', '1'))
            if hasattr(self, 'attention_type_var'):
                self.attention_type_var.set(arch_settings.get('attention_type', 'grouped_query'))
            if hasattr(self, 'query_groups_var'):
                self.query_groups_var.set(arch_settings.get('query_groups', '1'))
            
            # Load advanced settings
            adv_settings = settings.get('advanced_settings', {})
            if hasattr(self, 'use_checkpointing_var'):
                self.use_checkpointing_var.set(adv_settings.get('use_checkpointing', True))
            if hasattr(self, 'warmup_steps_entry'):
                self.warmup_steps_entry.delete(0, tk.END)
                self.warmup_steps_entry.insert(0, adv_settings.get('warmup_steps', '100'))
            if hasattr(self, 'auto_save_var'):
                self.auto_save_var.set(adv_settings.get('auto_save', True))
            if hasattr(self, 'auto_eval_var'):
                self.auto_eval_var.set(adv_settings.get('auto_eval', True))
            
            # Update dependent controls
            if hasattr(self, 'on_param_toggle'):
                self.on_param_toggle()
            if hasattr(self, 'on_attention_type_change'):
                self.on_attention_type_change()
                
            print("GUI settings loaded successfully")
            
        except Exception as e:
            print(f"Warning: Error loading GUI settings: {e}")

    def export_settings(self):
        """Export all settings including hyperparameters"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export All Settings As"
        )
        if filepath:
            try:
                # Update config from GUI first
                self.update_config_from_gui()
                
                settings = {
                    'hyperparameters': {
                        'learning_rate': self.config.learning_rate,
                        'weight_decay': self.config.weight_decay,
                        'n_embd': self.config.n_embd,
                        'n_layer': self.config.n_layer,
                        'n_head': self.config.n_head,
                        'block_size': self.config.block_size,
                        'batch_size': self.config.batch_size,
                        'dropout': self.config.dropout,
                        'attention_type': self.config.attention_type,
                        'n_query_groups': self.config.n_query_groups,
                        'use_moe': self.config.use_moe,
                        'moe_num_experts': self.config.moe_num_experts,
                        'moe_k': self.config.moe_k,
                        'vram_profile': getattr(self.config, 'vram_profile', 'low'),
                        'use_gradient_checkpointing': getattr(self.config, 'use_gradient_checkpointing', True),
                        'warmup_iters': self.config.warmup_iters,
                        'grad_clip': self.config.grad_clip,
                        'max_iters': self.config.max_iters
                    },
                    'gui_settings': {
                        'generation_settings': {
                            'max_tokens': self.max_tokens_entry.get(),
                            'temperature': self.temperature_entry.get(),
                            'top_k': self.top_k_entry.get(),
                            'top_p': self.top_p_entry.get(),
                            'use_top_k': self.use_top_k_var.get(),
                            'use_top_p': self.use_top_p_var.get(),
                            'generation_mode': self.generation_mode.get(),
                            'beam_size': self.beam_size_entry.get()
                        },
                        'training_settings': {
                            'iterations': self.iters_input.get(),
                            'plot_step': self.plot_step_input.get(),
                            'eval_iters': self.eval_iters_input.get(),
                            'auto_scroll': self.auto_scroll_var.get(),
                            'vram_profile': self.vram_profile_var.get()
                        },
                        'optimization_settings': {
                            'trials': self.trials_input.get(),
                            'steps_per_trial': self.steps_per_trial_input.get(),
                            'sampler': self.sampler_var.get(),
                            'pruner': self.pruner_var.get()
                        }
                    },
                    'export_info': {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'version': __version__,
                        'device': str(device)
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(settings, f, indent=4)
                
                messagebox.showinfo("Success", f"All settings exported to {os.path.basename(filepath)}")
                self.status_text.set(f"Settings exported to {os.path.basename(filepath)}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export settings:\n{e}")

    def import_settings(self):
        """Import all settings including hyperparameters"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Settings"
        )
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    settings = json.load(f)
                
                # Load hyperparameters if present
                if 'hyperparameters' in settings:
                    hyperparams = settings['hyperparameters']
                    for key, value in hyperparams.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    self.update_gui_from_config()
                
                # Load GUI settings if present
                if 'gui_settings' in settings:
                    # Save to temporary settings file and reload
                    temp_settings = settings['gui_settings']
                    with open(self.settings_file, 'w') as f:
                        json.dump(temp_settings, f, indent=4)
                    self.load_gui_settings()
                
                messagebox.showinfo("Success", f"Settings imported from {os.path.basename(filepath)}")
                self.status_text.set(f"Settings imported from {os.path.basename(filepath)}")
                
                # Ask if user wants to apply the imported hyperparameters
                if 'hyperparameters' in settings:
                    if messagebox.askyesno("Apply Hyperparameters", "Would you like to apply the imported hyperparameters to the model?"):
                        self.apply_hyperparameters()
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import settings:\n{e}")
        elif filepath:
            messagebox.showerror("File Not Found", f"The file '{filepath}' does not exist.")

    def cleanup_resources(self):
        """Cleanup resources and stop all running threads"""
        try:
            # Save settings before closing
            self.save_gui_settings()
            
            # Stop training if running
            if hasattr(self, 'stop_training_event') and self.stop_training_event:
                self.stop_training_event.set()
            
            # Stop optimization if running
            if hasattr(self, 'stop_optimization_event') and self.stop_optimization_event:
                self.stop_optimization_event.set()
            
            # Clear memory if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def save_model(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            title="Save Model As"
        )
        if filepath:
            self.trainer.save_model(filepath, verbose=True)
            self.status_text.set(f"Model saved to {os.path.basename(filepath)}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            title="Load Model"
        )
        if filepath and os.path.exists(filepath):
            try:
                # Use the enhanced model loading method from Trainer
                success = self.trainer.load_model(filepath)
                if success:
                    self.status_text.set(f"Model loaded from {os.path.basename(filepath)}")
                else:
                    self.status_text.set("Failed to load model - using current model.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model weights:\n{e}")
                self.status_text.set("Failed to load model.")
        elif filepath:
            messagebox.showerror("File Not Found", f"The file '{filepath}' does not exist.")

    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            self.cleanup_resources()
            # Give threads a moment to stop gracefully
            import time
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Error during shutdown: {e}")
        finally:
            self.root.destroy()

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
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text='Training')
        
        # Title and version info
        title_frame = tk.Frame(tab)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(title_frame, text=f"DGS-GPT v{__version__}", font=("Arial", 16, "bold"))
        title_label.pack(side='left')
        
        version_info = get_version_info()
        # Use a safer approach to get build info
        build_info = version_info.get('build_number', version_info.get('version', 'Unknown'))
        info_label = tk.Label(title_frame, text=f"Build: {build_info}", font=("Arial", 8))
        info_label.pack(side='right')
        
        # Create main horizontal container for console (left) and controls/plots (right)
        main_container = tk.Frame(tab)
        main_container.pack(fill='both', expand=True, pady=5)
        
        # Left side - Console (spanning full height)
        console_frame = tk.LabelFrame(main_container, text="Training Console", padx=10, pady=5)
        console_frame.pack(side='left', fill='both', expand=False, padx=(0,5))
        console_frame.config(width=350)  # Fixed width for console
        console_frame.pack_propagate(False)  # Prevent resizing based on content
        
        # Console controls
        console_controls = tk.Frame(console_frame)
        console_controls.pack(fill='x', pady=2)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(console_controls, text="Auto-scroll", variable=self.auto_scroll_var).pack(side='left', padx=5)
        
        tk.Button(console_controls, text="Clear Console", command=self.clear_console).pack(side='left', padx=5)
        
        tk.Button(console_controls, text="Save Log", command=self.save_console_log).pack(side='left', padx=5)
        
        # Console text area (full height)
        self.console_text = scrolledtext.ScrolledText(console_frame, width=40, wrap=tk.WORD, 
                                                     bg='black', fg='lime', font=('Consolas', 9))
        self.console_text.pack(fill='both', expand=True, pady=2)
        self.console_text.insert(tk.END, "=== DGS-GPT Training Console ===\n")
        import datetime
        self.console_text.insert(tk.END, f"Initialized at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.console_text.insert(tk.END, f"PyTorch Device: {device}\n")
        self.console_text.insert(tk.END, "Ready for training...\n\n")
        self.console_text.config(state='disabled')
        
        # Right side - Controls and plots
        right_container = tk.Frame(main_container)
        right_container.pack(side='right', fill='both', expand=True)
        
        # Controls frame with evaluation
        controls_frame = tk.Frame(right_container)
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
        
        tk.Label(controls_row1, text="Batch Size:").pack(side='left', padx=5)
        self.batch_size_input = tk.Entry(controls_row1, width=8)
        self.batch_size_input.insert(0, str(self.config.batch_size))
        self.batch_size_input.pack(side='left', padx=5)

        self.train_button = tk.Button(controls_row1, text="Start Training", command=self.start_training_thread)
        self.train_button.pack(side='left', padx=5)

        self.stop_button = tk.Button(controls_row1, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.pack(side='left', padx=5)
        
        # Second row - evaluation controls
        controls_row2 = tk.Frame(controls_frame)
        controls_row2.pack(fill='x', pady=2)
        
        self.eval_button = tk.Button(controls_row2, text="Evaluate Model", command=self.evaluate_model)
        self.eval_button.pack(side='left', padx=5)
        
        # Performance optimization controls (NEW)
        self.speed_config_button = tk.Button(controls_row2, text="🚀 Ultra-Speed Config", 
                                           command=self.apply_speed_config, 
                                           bg='orange', fg='white')
        self.speed_config_button.pack(side='left', padx=5)
        
        self.optimize_button = tk.Button(controls_row2, text="⚡ Optimize Now", 
                                       command=self.optimize_performance,
                                       bg='green', fg='white')
        self.optimize_button.pack(side='left', padx=5)
        
        self.clear_cache_button = tk.Button(controls_row2, text="🧹 Clear Cache", 
                                          command=self.clear_performance_cache,
                                          bg='blue', fg='white')
        self.clear_cache_button.pack(side='left', padx=5)
        
        tk.Label(controls_row2, text="Eval Iters:").pack(side='left', padx=5)
        self.eval_iters_input = tk.Entry(controls_row2, width=8)
        self.eval_iters_input.insert(0, "100")
        self.eval_iters_input.pack(side='left', padx=5)
        
        # Evaluation results
        self.eval_results_label = tk.Label(controls_row2, text="", fg="blue")
        self.eval_results_label.pack(side='left', padx=20)

        # Progress bar
        self.progress_bar = ttk.Progressbar(right_container, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)

        # Current hyperparameters display
        hyperparam_display_frame = tk.LabelFrame(right_container, text="Current Hyperparameters", padx=10, pady=5)
        hyperparam_display_frame.pack(fill='x', pady=5)
        
        self.hyperparam_text = scrolledtext.ScrolledText(hyperparam_display_frame, height=6, wrap=tk.WORD, state="disabled")
        self.hyperparam_text.pack(fill='x', pady=5)
        self.update_hyperparameter_display()

        self.fig = Figure(figsize=(14, 10), dpi=100, facecolor='white')
        
        # Initialize smoothing parameters
        self.smoothing_window = 10
        self.max_perplexity_display = 500  # Cap perplexity display
        
        # Create subplots with improved spacing
        self.ax1 = self.fig.add_subplot(221)  # Top left: Loss
        self.ax1.set_title("Training Loss", fontsize=12, fontweight='bold')
        self.ax1.set_xlabel("Iteration", fontsize=10)
        self.ax1.set_ylabel("Loss", fontsize=10)
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.loss_line, = self.ax1.plot([], [], 'r-', label='Loss', linewidth=1.5, alpha=0.7)
        self.loss_smooth_line, = self.ax1.plot([], [], 'r-', label='Loss (Smoothed)', linewidth=2.5)
        self.ax1.legend(fontsize=9)
        
        # Top right: Perplexity with improved scaling
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title("Training Perplexity", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Iteration", fontsize=10)
        self.ax2.set_ylabel("Perplexity", fontsize=10)
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.perplexity_line, = self.ax2.plot([], [], 'b-', label='Perplexity', linewidth=1.5, alpha=0.7)
        self.perplexity_smooth_line, = self.ax2.plot([], [], 'b-', label='Perplexity (Smoothed)', linewidth=2.5)
        self.ax2.legend(fontsize=9)
        
        # Bottom left: Combined view with dual y-axis (improved)
        self.ax3 = self.fig.add_subplot(223)
        self.ax3.set_title("Loss & Perplexity Combined", fontsize=12, fontweight='bold')
        self.ax3.set_xlabel("Iteration", fontsize=10)
        self.ax3.set_ylabel("Loss", color='darkred', fontsize=10)
        self.ax3.tick_params(axis='y', labelcolor='darkred')
        self.ax3.grid(True, alpha=0.3, linestyle='--')
        self.combined_loss_line, = self.ax3.plot([], [], 'r-', label='Loss', linewidth=2)
        
        # Create second y-axis for perplexity with better scaling
        self.ax3_twin = self.ax3.twinx()
        self.ax3_twin.set_ylabel("Perplexity", color='darkblue', fontsize=10)
        self.ax3_twin.tick_params(axis='y', labelcolor='darkblue')
        self.combined_perplexity_line, = self.ax3_twin.plot([], [], 'b-', label='Perplexity', linewidth=2)
        
        # Add legends for combined plot
        lines1, labels1 = self.ax3.get_legend_handles_labels()
        lines2, labels2 = self.ax3_twin.get_legend_handles_labels()
        self.ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        # Bottom right: Learning rate and metrics
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.set_title("Learning Rate & Gradient Norm", fontsize=12, fontweight='bold')
        self.ax4.set_xlabel("Iteration", fontsize=10)
        self.ax4.set_ylabel("Learning Rate", color='green', fontsize=10)
        self.ax4.tick_params(axis='y', labelcolor='green')
        self.ax4.grid(True, alpha=0.3, linestyle='--')
        self.lr_line, = self.ax4.plot([], [], 'g-', label='Learning Rate', linewidth=2)
        
        # Set initial y-axis limits for learning rate to avoid empty plot
        self.ax4.set_ylim(0, 0.001)  # Default range
        
        # Second y-axis for gradient norm
        self.ax4_twin = self.ax4.twinx()
        self.ax4_twin.set_ylabel("Gradient Norm", color='orange', fontsize=10)
        self.ax4_twin.tick_params(axis='y', labelcolor='orange')
        self.grad_norm_line, = self.ax4_twin.plot([], [], 'orange', label='Grad Norm', linewidth=2)
        
        # Set initial y-axis limits for gradient norm
        self.ax4_twin.set_ylim(0, 2.0)  # Default range
        
        # Add legends for metrics plot
        lines3, labels3 = self.ax4.get_legend_handles_labels()
        lines4, labels4 = self.ax4_twin.get_legend_handles_labels()
        self.ax4.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=9)
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout(pad=4.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_model_settings_tab(self):
        """Create unified Model Settings tab combining VRAM profiles and hyperparameters"""
        settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(settings_frame, text='Model Settings')

        # Create scrollable frame for all controls
        canvas = tk.Canvas(settings_frame)
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # VRAM Optimization Section
        vram_frame = tk.LabelFrame(scrollable_frame, text="VRAM & Memory Settings", padx=10, pady=5)
        vram_frame.pack(fill='x', pady=5)
        
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

        # Quick Apply Section
        apply_frame = tk.LabelFrame(scrollable_frame, text="Quick Apply to Training", padx=10, pady=5)
        apply_frame.pack(fill='x', pady=5)
        
        apply_controls = tk.Frame(apply_frame)
        apply_controls.pack(fill='x', pady=5)
        
        tk.Button(apply_controls, text="Apply Current Settings", command=self.apply_current_settings,
                 bg='lightgreen', font=("Arial", 10, "bold")).pack(side='left', padx=5)
        tk.Button(apply_controls, text="Reset to Defaults", command=self.reset_model_settings,
                 bg='lightcoral').pack(side='left', padx=5)
        
        # Apply settings info
        tk.Label(apply_frame, text="Apply changes to active training session without restarting",
                font=("Arial", 8), fg="gray").pack(anchor='w', padx=5)

        # Architecture Settings
        arch_frame = tk.LabelFrame(scrollable_frame, text="Model Architecture", padx=10, pady=5)
        arch_frame.pack(fill='x', pady=5)
        
        # Basic model parameters
        basic_arch = tk.Frame(arch_frame)
        basic_arch.pack(fill='x', pady=2)
        
        tk.Label(basic_arch, text="Embedding Dim:").pack(side='left', padx=5)
        self.embd_dim_var = tk.StringVar(value=str(self.config.n_embd))
        embd_combo = ttk.Combobox(basic_arch, textvariable=self.embd_dim_var, 
                                 values=["256", "512", "1024", "1536", "2048"], width=8, state="readonly")
        embd_combo.pack(side='left', padx=5)
        
        tk.Label(basic_arch, text="Layers:").pack(side='left', padx=(20,5))
        self.layers_var = tk.StringVar(value=str(self.config.n_layer))
        layers_combo = ttk.Combobox(basic_arch, textvariable=self.layers_var, 
                                   values=["4", "6", "8", "12", "16", "20", "24"], width=8, state="readonly")
        layers_combo.pack(side='left', padx=5)
        
        tk.Label(basic_arch, text="Heads:").pack(side='left', padx=(20,5))
        self.heads_var = tk.StringVar(value=str(self.config.n_head))
        heads_combo = ttk.Combobox(basic_arch, textvariable=self.heads_var, 
                                  values=["4", "8", "12", "16", "20"], width=8, state="readonly")
        heads_combo.pack(side='left', padx=5)

        # Advanced architecture
        adv_arch = tk.Frame(arch_frame)
        adv_arch.pack(fill='x', pady=2)
        
        self.use_moe_var = tk.BooleanVar(value=self.config.use_moe)
        tk.Checkbutton(adv_arch, text="Use Mixture of Experts (MoE)", variable=self.use_moe_var).pack(side='left', padx=5)
        
        tk.Label(adv_arch, text="Experts:").pack(side='left', padx=(20,5))
        self.moe_experts_var = tk.StringVar(value=str(self.config.moe_num_experts))
        moe_experts_combo = ttk.Combobox(adv_arch, textvariable=self.moe_experts_var, 
                                        values=["2", "4", "8"], width=5, state="readonly")
        moe_experts_combo.pack(side='left', padx=5)
        
        tk.Label(adv_arch, text="Top-K:").pack(side='left', padx=(10,5))
        self.moe_k_var = tk.StringVar(value=str(self.config.moe_k))
        moe_k_combo = ttk.Combobox(adv_arch, textvariable=self.moe_k_var, 
                                  values=["1", "2"], width=5, state="readonly")
        moe_k_combo.pack(side='left', padx=5)

        # Attention settings
        attn_arch = tk.Frame(arch_frame)
        attn_arch.pack(fill='x', pady=2)
        
        tk.Label(attn_arch, text="Attention Type:").pack(side='left', padx=5)
        self.attention_type_var = tk.StringVar(value=self.config.attention_type)
        attn_combo = ttk.Combobox(attn_arch, textvariable=self.attention_type_var, 
                                 values=["multihead", "grouped_query"], width=15, state="readonly")
        attn_combo.pack(side='left', padx=5)
        attn_combo.bind('<<ComboboxSelected>>', self.on_attention_type_change)
        
        tk.Label(attn_arch, text="Query Groups:").pack(side='left', padx=(20,5))
        self.query_groups_var = tk.StringVar(value=str(self.config.n_query_groups))
        self.query_groups_combo = ttk.Combobox(attn_arch, textvariable=self.query_groups_var, 
                                              values=["1", "2", "4", "8"], width=5, state="readonly")
        self.query_groups_combo.pack(side='left', padx=5)

        # Training Hyperparameters
        hyper_frame = tk.LabelFrame(scrollable_frame, text="Training Hyperparameters", padx=10, pady=5)
        hyper_frame.pack(fill='x', pady=5)
        
        # Learning parameters
        lr_frame = tk.Frame(hyper_frame)
        lr_frame.pack(fill='x', pady=2)
        
        tk.Label(lr_frame, text="Learning Rate:").pack(side='left', padx=5)
        self.lr_var = tk.StringVar(value=str(self.config.learning_rate))
        lr_entry = tk.Entry(lr_frame, textvariable=self.lr_var, width=12)
        lr_entry.pack(side='left', padx=5)
        
        tk.Label(lr_frame, text="Weight Decay:").pack(side='left', padx=(20,5))
        self.wd_var = tk.StringVar(value=str(self.config.weight_decay))
        wd_entry = tk.Entry(lr_frame, textvariable=self.wd_var, width=12)
        wd_entry.pack(side='left', padx=5)
        
        tk.Label(lr_frame, text="Dropout:").pack(side='left', padx=(20,5))
        self.dropout_var = tk.StringVar(value=str(self.config.dropout))
        dropout_entry = tk.Entry(lr_frame, textvariable=self.dropout_var, width=12)
        dropout_entry.pack(side='left', padx=5)

        # Batch and optimization
        batch_frame = tk.Frame(hyper_frame)
        batch_frame.pack(fill='x', pady=2)
        
        tk.Label(batch_frame, text="Batch Size:").pack(side='left', padx=5)
        self.batch_size_var = tk.StringVar(value=str(self.config.batch_size))
        batch_entry = tk.Entry(batch_frame, textvariable=self.batch_size_var, width=8)
        batch_entry.pack(side='left', padx=5)
        
        tk.Label(batch_frame, text="Block Size:").pack(side='left', padx=(20,5))
        self.block_size_var = tk.StringVar(value=str(self.config.block_size))
        block_combo = ttk.Combobox(batch_frame, textvariable=self.block_size_var, 
                                  values=["512", "1024", "2048", "4096"], width=8, state="readonly")
        block_combo.pack(side='left', padx=5)

        # Optimization Settings
        optim_frame = tk.LabelFrame(scrollable_frame, text="Hyperparameter Optimization", padx=10, pady=5)
        optim_frame.pack(fill='x', pady=5)
        
        # Basic optimization controls
        optim_basic = tk.Frame(optim_frame)
        optim_basic.pack(fill='x', pady=2)
        
        tk.Label(optim_basic, text="Trials:").pack(side='left', padx=5)
        self.trials_input = tk.Entry(optim_basic, width=8)
        self.trials_input.insert(0, "10")
        self.trials_input.pack(side='left', padx=5)
        
        tk.Label(optim_basic, text="Steps/Trial:").pack(side='left', padx=(20,5))
        self.steps_per_trial_input = tk.Entry(optim_basic, width=8)
        self.steps_per_trial_input.insert(0, "100")
        self.steps_per_trial_input.pack(side='left', padx=5)
        
        tk.Label(optim_basic, text="Sampler:").pack(side='left', padx=(20,5))
        self.sampler_var = tk.StringVar(value="tpe")
        sampler_combo = ttk.Combobox(optim_basic, textvariable=self.sampler_var, 
                                    values=["tpe", "random", "grid", "cmaes"], width=10, state="readonly")
        sampler_combo.pack(side='left', padx=5)

        # Optimization checkboxes
        optim_params = tk.Frame(optim_frame)
        optim_params.pack(fill='x', pady=5)
        
        tk.Label(optim_params, text="Optimize:", font=("Arial", 9, "bold")).pack(anchor='w')
        
        checkbox_frame = tk.Frame(optim_params)
        checkbox_frame.pack(fill='x')
        
        self.optimize_lr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(checkbox_frame, text="Learning Rate", variable=self.optimize_lr_var).pack(side='left', padx=10)
        
        self.optimize_wd_var = tk.BooleanVar(value=True)
        tk.Checkbutton(checkbox_frame, text="Weight Decay", variable=self.optimize_wd_var).pack(side='left', padx=10)
        
        self.optimize_embd_var = tk.BooleanVar(value=True)
        tk.Checkbutton(checkbox_frame, text="Embedding Dim", variable=self.optimize_embd_var).pack(side='left', padx=10)
        
        self.optimize_layers_var = tk.BooleanVar(value=True)
        tk.Checkbutton(checkbox_frame, text="Layers", variable=self.optimize_layers_var).pack(side='left', padx=10)

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
        self.optim_log_text = scrolledtext.ScrolledText(scrollable_frame, height=10, wrap=tk.WORD, state="disabled")
        self.optim_log_text.pack(fill="both", expand=True, pady=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Initialize VRAM profile info
        if not (hasattr(self, '_initializing') and self._initializing):
            self.on_vram_profile_change()
        
        # Update attention type state
        self.on_attention_type_change()

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
        # Only process if we have the advanced optimization UI (which we don't in the current simplified interface)
        if not hasattr(self, 'lr_min_entry'):
            # Skip this method if we don't have the advanced optimization interface
            return
        
        # This method is for advanced optimization UI that doesn't exist in current interface
        pass

    def get_parameter_config(self):
        """Get parameter configuration from GUI controls"""
        param_config = {}
        
        # Only process if we have the advanced optimization UI (which we don't in the current simplified interface)
        if not hasattr(self, 'lr_min_entry'):
            # Return basic parameter config from current GUI elements
            param_config = {
                'optimize_lr': getattr(self, 'optimize_lr_var', tk.BooleanVar(value=True)).get(),
                'optimize_wd': getattr(self, 'optimize_wd_var', tk.BooleanVar(value=True)).get(),
                'optimize_embd': getattr(self, 'optimize_embd_var', tk.BooleanVar(value=True)).get(),
                'optimize_layers': getattr(self, 'optimize_layers_var', tk.BooleanVar(value=True)).get(),
                'optimize_heads': getattr(self, 'optimize_heads_var', tk.BooleanVar(value=False)).get(),
                'optimize_dropout': getattr(self, 'optimize_dropout_var', tk.BooleanVar(value=True)).get(),
                'optimize_batch': getattr(self, 'optimize_batch_var', tk.BooleanVar(value=False)).get(),
            }
            return param_config
        
        # Advanced parameter configuration (not used in current interface)
        return {}

    def get_sampler_config(self):
        """Get sampler configuration from GUI controls"""
        sampler_config = {}
        
        sampler_config['type'] = self.sampler_var.get()
        sampler_config['pruner'] = self.pruner_var.get()
        
        try:
            sampler_config['n_startup_trials'] = int(self.startup_trials_entry.get())
        except (ValueError, AttributeError):
            sampler_config['n_startup_trials'] = 10
        
        seed_text = self.seed_entry.get().strip()
        if seed_text:
            try:
                sampler_config['seed'] = int(seed_text)
            except (ValueError, TypeError):
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
        self.loss_queue.put(('loss', loss))
    
    def queue_metrics(self, metrics):
        self.loss_queue.put(('metrics', metrics))

    def clear_console(self):
        """Clear the console log"""
        self.console_text.config(state='normal')
        self.console_text.delete(1.0, tk.END)
        self.console_text.insert(tk.END, "=== Console Cleared ===\n")
        self.console_text.config(state='disabled')
    
    def save_console_log(self):
        """Save console log to file"""
        from tkinter import filedialog
        import datetime
        
        default_filename = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialname=default_filename
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.console_text.get(1.0, tk.END))
                messagebox.showinfo("Log Saved", f"Console log saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log:\n{str(e)}")
    
    def log_to_console(self, message, level="INFO"):
        """Add a message to the console log"""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Color coding for different log levels
        color_map = {
            "INFO": "lime",
            "WARNING": "yellow", 
            "ERROR": "red",
            "SUCCESS": "lightgreen",
            "DEBUG": "cyan"
        }
        
        self.console_text.config(state='normal')
        
        # Add timestamp and level
        self.console_text.insert(tk.END, f"[{timestamp}] {level}: ", )
        
        # Add the message
        self.console_text.insert(tk.END, f"{message}\n")
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.console_text.see(tk.END)
        
        self.console_text.config(state='disabled')

    def log_error(self, error_msg, context="Unknown"):
        """Enhanced error logging with context"""
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] ERROR in {context}: {error_msg}"
        self.log_to_console(formatted_msg, "ERROR")
        print(formatted_msg)  # Also print to console for debugging

    def smooth_data(self, data, window_size=None):
        """Apply exponential moving average smoothing to data"""
        if not data or len(data) < 2:
            return data
        
        if window_size is None:
            window_size = self.smoothing_window
            
        smoothed = [data[0]]  # Start with first value
        alpha = 2.0 / (window_size + 1)  # EMA smoothing factor
        
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
        
        return smoothed

    def process_loss_queue(self):
        try:
            # Process items from queue in batches to reduce overhead
            queue_items_processed = 0
            plot_update_needed = False
            
            while not self.loss_queue.empty() and queue_items_processed < 10:  # Limit batch size
                data_type, data = self.loss_queue.get_nowait()
                queue_items_processed += 1
                plot_update_needed = True
                
                if data_type == 'loss':
                    # Legacy loss callback for backward compatibility
                    self.losses.append(data)
                    
                    # Generate sequential iteration numbers for legacy mode
                    next_iteration = len(self.losses) - 1
                    self.iterations.append(next_iteration)
                    
                    # Calculate perplexity from loss if metrics not available
                    perplexity = math.exp(min(data, 20)) if data < 20 else self.max_perplexity_display
                    self.perplexities.append(perplexity)
                    
                    # Add default values for missing metrics
                    self.learning_rates.append(0.0)
                    self.grad_norms.append(0.0)
                        
                elif data_type == 'metrics':
                    # New comprehensive metrics - always append to maintain order
                    iteration = data['iteration']
                    loss = data['loss']
                    perplexity = data.get('perplexity', math.exp(min(loss, 20)) if loss < 20 else self.max_perplexity_display)
                    learning_rate = data.get('learning_rate', 0.0)
                    grad_norm = data.get('grad_norm', 0.0)
                    
                    # Cap perplexity to reasonable values for visualization
                    if math.isinf(perplexity) or math.isnan(perplexity) or perplexity > self.max_perplexity_display:
                        perplexity = self.max_perplexity_display
                    
                    # Always append to maintain chronological order and prevent x-axis jumping
                    # Ensure iteration is monotonically increasing
                    if len(self.iterations) > 0:
                        last_iter = self.iterations[-1]
                        if iteration <= last_iter:
                            iteration = last_iter + 1
                    
                    self.iterations.append(iteration)
                    self.losses.append(loss)
                    self.perplexities.append(perplexity)
                    self.learning_rates.append(learning_rate)
                    self.grad_norms.append(grad_norm)
            
            # Only update plot and sync arrays if needed and at reduced frequency
            if plot_update_needed and len(self.losses) % 5 == 0:  # Update plot every 5 data points instead of every point
                self.validate_and_sync_data_arrays()
                self.update_plot()
                
        except Exception as e:
            self.log_to_console(f"Error processing loss queue: {str(e)}", "ERROR")
        finally:
            # Increase update interval to 200ms to reduce CPU overhead
            self.root.after(200, self.process_loss_queue)

    def validate_and_sync_data_arrays(self):
        """Ensure all data arrays are properly synchronized and monotonic"""
        if not self.losses:
            return
            
        # Limit data length to prevent memory issues (keep last 2000 points)
        max_data_points = 2000
        if len(self.losses) > max_data_points:
            # Trim all arrays to keep only the most recent data
            trim_count = len(self.losses) - max_data_points
            self.losses = self.losses[trim_count:]
            self.iterations = self.iterations[trim_count:] if self.iterations else []
            self.perplexities = self.perplexities[trim_count:] if self.perplexities else []
            self.learning_rates = self.learning_rates[trim_count:] if self.learning_rates else []
            self.grad_norms = self.grad_norms[trim_count:] if self.grad_norms else []
            
        # Get target length from the primary array (losses)
        target_length = len(self.losses)
        
        # Ensure all arrays have the same length
        arrays = {
            'iterations': self.iterations,
            'perplexities': self.perplexities, 
            'learning_rates': self.learning_rates,
            'grad_norms': self.grad_norms
        }
        
        for name, array in arrays.items():
            while len(array) < target_length:
                if name == 'iterations':
                    # For iterations, use sequential numbering
                    next_val = len(array)
                    if len(array) > 0:
                        next_val = max(array[-1] + 1, len(array))
                    array.append(next_val)
                elif name == 'perplexities':
                    # For perplexity, calculate from corresponding loss
                    loss_idx = len(array)
                    if loss_idx < len(self.losses):
                        loss = self.losses[loss_idx]
                        perp = math.exp(min(loss, 20)) if loss < 20 else self.max_perplexity_display
                        array.append(perp)
                    else:
                        array.append(self.max_perplexity_display)
                else:
                    # For other metrics, use default values
                    array.append(0.0)
            
            # Trim if too long
            if len(array) > target_length:
                arrays[name] = array[:target_length]
        
        # Update the instance variables
        self.iterations = arrays['iterations']
        self.perplexities = arrays['perplexities']
        self.learning_rates = arrays['learning_rates']
        self.grad_norms = arrays['grad_norms']
        
        # Ensure iterations are strictly monotonically increasing
        if len(self.iterations) > 1:
            for i in range(1, len(self.iterations)):
                if self.iterations[i] <= self.iterations[i-1]:
                    self.iterations[i] = self.iterations[i-1] + 1

    def update_plot(self):
        if not self.losses:
            return
            
        # Skip plot updates if data hasn't changed significantly
        if hasattr(self, '_last_plot_size') and len(self.losses) - self._last_plot_size < 3:
            return
        self._last_plot_size = len(self.losses)
            
        # Ensure data synchronization before plotting
        self.validate_and_sync_data_arrays()
        
        # Use iterations as x-axis data (should now be monotonic)
        x_data = self.iterations if self.iterations else list(range(len(self.losses)))
        
        # --- Update Loss Plot (Top Left) ---
        if hasattr(self, 'loss_line'):
            self.loss_line.set_data(x_data, self.losses)
            if len(self.losses) >= 3 and hasattr(self, 'loss_smooth_line'):
                smoothed_losses = self.smooth_data(self.losses)
                self.loss_smooth_line.set_data(x_data, smoothed_losses)
            
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Set reasonable y-limits for loss with caching
            if self.losses and (not hasattr(self, '_cached_loss_limits') or len(self.losses) % 20 == 0):
                min_loss = min(self.losses)
                max_loss = max(self.losses)
                if max_loss > min_loss:
                    margin = (max_loss - min_loss) * 0.1
                    self._cached_loss_limits = (max(0, min_loss - margin), max_loss + margin)
                    self.ax1.set_ylim(*self._cached_loss_limits)
        
        # --- Update Perplexity Plot (Top Right) with reduced frequency ---
        if hasattr(self, 'perplexity_line') and self.perplexities and len(self.losses) % 2 == 0:
            # Filter valid perplexity values
            valid_perplexities = []
            valid_x = []
            for i, (x, p) in enumerate(zip(x_data, self.perplexities)):
                if not math.isinf(p) and not math.isnan(p) and p <= self.max_perplexity_display:
                    valid_perplexities.append(p)
                    valid_x.append(x)
            
            if valid_perplexities:
                self.perplexity_line.set_data(valid_x, valid_perplexities)
                if len(valid_perplexities) >= 3 and hasattr(self, 'perplexity_smooth_line'):
                    smoothed_perplexities = self.smooth_data(valid_perplexities)
                    self.perplexity_smooth_line.set_data(valid_x, smoothed_perplexities)
            
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        # --- Update Combined Plot (Bottom Left) with reduced frequency ---
        if hasattr(self, 'combined_loss_line') and len(self.losses) % 3 == 0:
            self.combined_loss_line.set_data(x_data, self.losses)
            
            if hasattr(self, 'combined_perplexity_line') and self.perplexities:
                valid_perplexities = [min(p, self.max_perplexity_display) for p in self.perplexities 
                                    if not math.isinf(p) and not math.isnan(p)]
                if valid_perplexities and len(valid_perplexities) == len(x_data):
                    self.combined_perplexity_line.set_data(x_data, valid_perplexities)
            
            self.ax3.relim()
            self.ax3.autoscale_view()
            if hasattr(self, 'ax3_twin'):
                self.ax3_twin.relim()
                self.ax3_twin.autoscale_view()
        
        # --- Update Learning Rate and Gradient Norm Plot (Bottom Right) ---
        if hasattr(self, 'lr_line') and self.learning_rates and len(self.losses) % 4 == 0:
            # Ensure data arrays match x_data length
            lr_data = self.learning_rates[:len(x_data)]
            lr_x_data = x_data[:len(lr_data)]
            
            if lr_data and lr_x_data and len(lr_data) == len(lr_x_data):
                self.lr_line.set_data(lr_x_data, lr_data)
                self.ax4.relim()
                self.ax4.autoscale_view()
        
        if hasattr(self, 'grad_norm_line') and self.grad_norms and len(self.losses) % 4 == 0:
            # Ensure data arrays match x_data length and filter extreme values
            gn_data = self.grad_norms[:len(x_data)]
            filtered_grad_norms = [min(max(gn, 0.001), 10.0) for gn in gn_data]  # Cap between 0.001 and 10
            gn_x_data = x_data[:len(filtered_grad_norms)]
            
            if filtered_grad_norms and gn_x_data and len(filtered_grad_norms) == len(gn_x_data):
                self.grad_norm_line.set_data(gn_x_data, filtered_grad_norms)
                if hasattr(self, 'ax4_twin'):
                    self.ax4_twin.relim()
                    self.ax4_twin.autoscale_view()
        
        # Redraw the canvas with optimized method
        try:
            if hasattr(self, 'fig') and hasattr(self.fig, 'canvas'):
                # Use draw_idle for much better performance - only redraws when GUI is idle
                self.fig.canvas.draw_idle()
        except Exception as e:
            self.log_to_console(f"Plot draw error: {str(e)}", "ERROR")

    def start_training_thread(self):
        try:
            max_iters = int(self.iters_input.get())
            plot_step_size = int(self.plot_step_input.get())
            batch_size = int(self.batch_size_input.get())
            if max_iters <= 0 or plot_step_size <= 0 or batch_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive integers for iterations, plot step, and batch size.")
            return

        # Update config with the new batch size and recreate trainer if needed
        if self.config.batch_size != batch_size:
            self.config.batch_size = batch_size
            
            # Recreate trainer with updated batch size to ensure proper data loading
            try:
                old_trainer = self.trainer
                self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
                
                # Try to transfer model weights if compatible
                if hasattr(old_trainer, 'model') and hasattr(self.trainer, 'model'):
                    try:
                        old_state = old_trainer.model.state_dict()
                        new_state = self.trainer.model.state_dict()
                        
                        # Transfer compatible weights
                        compatible_keys = []
                        for key in old_state:
                            if key in new_state and old_state[key].shape == new_state[key].shape:
                                new_state[key] = old_state[key]
                                compatible_keys.append(key)
                        
                        if compatible_keys:
                            self.trainer.model.load_state_dict(new_state)
                            self.log_to_console(f"Transferred {len(compatible_keys)} compatible layers to new trainer", "INFO")
                        else:
                            self.log_to_console("No compatible weights to transfer - using fresh model", "WARNING")
                    except Exception as e:
                        self.log_to_console(f"Could not transfer weights: {e}", "WARNING")
                
                self.log_to_console(f"Trainer recreated with batch size: {batch_size}", "INFO")
                
            except Exception as e:
                self.log_to_console(f"Warning: Could not recreate trainer with new batch size: {e}", "WARNING")
                # Fall back to just updating the config
                if hasattr(self, 'trainer') and self.trainer:
                    self.trainer.config.batch_size = batch_size
        else:
            self.log_to_console(f"Using current batch size: {batch_size}", "INFO")

        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.generate_button.config(state="disabled")
        self.status_text.set(f"Training for {max_iters} iterations...")
        
        # Log training start
        self.log_to_console(f"Starting training for {max_iters} iterations (plot step: {plot_step_size})", "INFO")
        self.log_to_console(f"Model parameters: {sum(p.numel() for p in self.trainer.model.parameters()):,}", "INFO")
        
        # Clear previous data and reset plot
        self.clear_plot_data()
        
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = max_iters

        self.stop_training_event = threading.Event()
        thread = threading.Thread(target=self.run_training, args=(max_iters, plot_step_size))
        thread.daemon = True
        thread.start()

    def clear_plot_data(self):
        """Clear all plotting data arrays"""
        self.losses = []
        self.perplexities = []
        self.iterations = []
        self.learning_rates = []
        self.grad_norms = []
        
        # Clear the plot immediately
        self.update_plot()
        
        # Log the clearing
        self.log_to_console("Plot data cleared for new training session", "INFO")

    def run_training(self, max_iters, plot_step_size):
        try:
            self.trainer.train(max_iters, plot_step_size, self.stop_training_event, self.update_progress)
            if self.stop_training_event.is_set():
                self.root.after(0, lambda: self.status_text.set("Training stopped by user."))
                self.root.after(0, lambda: self.log_to_console("Training stopped by user", "WARNING"))
            else:
                self.root.after(0, lambda: self.status_text.set("Training complete. Ready."))
                self.root.after(0, lambda: self.log_to_console("Training completed successfully!", "SUCCESS"))
        except Exception as e:
            self.root.after(0, lambda error=e: self.status_text.set(f"Training error: {error}"))
            self.root.after(0, lambda error=e: self.log_to_console(f"Training error: {str(error)}", "ERROR"))
        finally:
            self.root.after(0, lambda: self.train_button.config(state="normal"))
            self.root.after(0, lambda: self.stop_button.config(state="disabled"))
            self.root.after(0, lambda: self.generate_button.config(state="normal"))

    def stop_training(self):
        if self.stop_training_event:
            self.stop_training_event.set()
            self.log_to_console("Training stop requested by user", "WARNING")
        self.stop_button.config(state="disabled")

    def update_progress(self, current_step):
        self.root.after(0, self._set_progress, current_step)

    def _set_progress(self, value):
        self.progress_bar['value'] = value
        
        # Log progress periodically
        if value % 100 == 0 and value > 0:
            progress_pct = (value / self.progress_bar['maximum']) * 100 if self.progress_bar['maximum'] > 0 else 0
            self.log_to_console(f"Training progress: {value}/{self.progress_bar['maximum']} ({progress_pct:.1f}%)", "INFO")

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
            self.root.after(0, lambda error=e: self.status_text.set(f"Error: {error}"))
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
                self.root.after(0, lambda error=e: self.status_text.set(f"Evaluation error: {error}"))
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
        # Skip during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
            
        # Prevent recursive calls
        if hasattr(self, '_updating_vram_profile') and self._updating_vram_profile:
            return
        
        profile = self.vram_profile_var.get()
        
        if profile == "low":
            info_text = "15GB VRAM: batch_size=2, context=512, layers=4, embedding=1024"
        elif profile == "medium":
            info_text = "40GB VRAM: batch_size=16, context=8192, layers=32, embedding=4096"
        elif profile == "high":
            info_text = "100GB+ VRAM: batch_size=32, context=16384, layers=40, embedding=5120"
        else:
            info_text = ""
            
        # Only update the label if it exists (GUI components are created)
        if hasattr(self, 'vram_info_label'):
            self.vram_info_label.config(text=info_text)
        
        # Log the VRAM profile change to console (only if console exists)
        if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
            self.log_to_console(f"VRAM Profile changed to: {profile.upper()}", "INFO")
            self.log_to_console(f"Profile settings: {info_text}", "INFO")
        
        # Update config if trainer exists
        if hasattr(self, 'trainer') and self.trainer:
            try:
                # Set flag to prevent recursive calls
                self._updating_vram_profile = True
                
                # Store old trainer reference before creating new one
                old_trainer = self.trainer
                
                # Apply VRAM optimized config
                print(f"Applying {profile.upper()} VRAM profile...")
                print(f"Current config batch size before: {self.config.batch_size}")
                
                # Pass None as base_config to force creation of a fresh config with VRAM optimizations
                new_config = Config.get_vram_optimized_config(profile, None)
                
                # Preserve important settings from the current config
                new_config.vocab_size = self.config.vocab_size
                if hasattr(self.config, 'device'):
                    new_config.device = self.config.device
                    
                print(f"New config batch size after: {new_config.batch_size}")
                self.config = new_config
                print(f"Config updated with {profile.upper()} profile - batch size: {self.config.batch_size}")
                
                # Recreate trainer with new config
                print("Creating new trainer with updated config...")
                
                # Clear GPU cache before creating new trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("GPU cache cleared")
                
                self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
                print("New trainer created successfully")
                
                # Try to transfer model weights if compatible
                if hasattr(old_trainer, 'model') and hasattr(self.trainer, 'model'):
                    try:
                        # Check if architectures are compatible (same basic structure)
                        old_state = old_trainer.model.state_dict()
                        new_state = self.trainer.model.state_dict()
                        
                        # Transfer compatible weights
                        compatible_weights = {}
                        for key in new_state.keys():
                            if key in old_state and old_state[key].shape == new_state[key].shape:
                                compatible_weights[key] = old_state[key]
                        
                        if compatible_weights:
                            self.trainer.model.load_state_dict(compatible_weights, strict=False)
                            print(f"Transferred {len(compatible_weights)} compatible weight tensors")
                        else:
                            print("No compatible weights to transfer - using fresh model")
                            
                    except Exception as weight_error:
                        print(f"Weight transfer failed: {weight_error}")
                        if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
                            self.log_to_console(f"Weight transfer failed: {weight_error}", "WARNING")
                
                # Update GUI fields with new config values
                print("Updating GUI from config...")
                self.update_gui_from_config()
                
                # Update displays only if GUI components exist
                if hasattr(self, 'hyperparam_text'):
                    self.update_hyperparameter_display()
                
                print(f"Applied {profile.upper()} VRAM profile - Config and GUI updated")
                
                # Log success to console if available
                if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
                    self.log_to_console(f"Successfully applied {profile.upper()} VRAM profile", "SUCCESS")
                    self.log_to_console(f"New settings - Batch: {self.config.batch_size}, Layers: {self.config.n_layer}, Embedding: {self.config.n_embd}", "INFO")
                    
            except Exception as e:
                error_msg = f"Error applying VRAM profile: {e}"
                print(error_msg)
                
                # Log error to console if available
                if hasattr(self, 'console_text') and hasattr(self, 'log_to_console'):
                    self.log_to_console(error_msg, "ERROR")
                    self.log_to_console("VRAM profile change failed - keeping previous configuration", "WARNING")
                
                # Show error dialog to user
                try:
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("VRAM Profile Error", 
                                       f"Failed to apply {profile.upper()} VRAM profile:\n{str(e)}\n\nKeeping previous configuration.")
                except:
                    pass  # Fallback if messagebox fails
                
                # Revert to previous profile in GUI if possible
                try:
                    if hasattr(self, '_last_working_profile'):
                        self.vram_profile_var.set(self._last_working_profile)
                except:
                    pass
                    
            finally:
                # Clear the flag
                self._updating_vram_profile = False
        else:
            # Store the working profile for potential revert
            self._last_working_profile = profile

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
        self.warmup_steps_entry.insert(0, "100")
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
        tk.Checkbutton(autosave_frame, text="Auto-save best model during training", 
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
            self.config.auto_save_best = self.auto_save_var.get()  # Add auto-save setting
            
            # Create new trainer with updated config
            self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
            
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
        self.optim_log_text.insert(tk.END, f"Device: {device}\n")
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
        """Run hyperparameter optimization in a separate thread"""
        try:
            # Use the optimization function from ShitGPT
            best_config = run_hyperparameter_optimization(
                arch_config, n_trials, steps_per_trial, 
                param_config, self.stop_optimization_event, sampler_config
            )
            
            # Update config with best parameters
            if best_config:
                self.config = best_config
                self.config.vocab_size = vocab_size
                
                # Update GUI with best parameters
                self.root.after(0, self.update_gui_from_config)
                self.root.after(0, self.update_hyperparameter_display)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Optimization Error", f"Error during optimization: {e}"))
        finally:
            self.root.after(0, self.optimization_finished)

    def process_optim_queue(self):
        """Process optimization log queue"""
        try:
            while not self.optim_queue.empty():
                message = self.optim_queue.get_nowait()
                self.optim_log_text.config(state="normal")
                self.optim_log_text.insert(tk.END, message + "\n")
                self.optim_log_text.see(tk.END)
                self.optim_log_text.config(state="disabled")
        except:
            pass
        
        if not (self.stop_optimization_event and self.stop_optimization_event.is_set()):
            self.root.after(100, self.process_optim_queue)

    def stop_optimization(self):
        """Stop optimization"""
        if self.stop_optimization_event:
            self.stop_optimization_event.set()
        self.stop_optimize_button.config(state="disabled")

    def optimization_finished(self):
        """Handle optimization completion"""
        self.optimize_button.config(state="normal")
        self.stop_optimize_button.config(state="disabled")
        self.train_button.config(state="normal")
        self.generate_button.config(state="normal")
        self.status_text.set("Optimization complete. Ready.")

    def load_model_weights(self):
        model_path = "gpt_model.pth"
        if os.path.exists(model_path):
            try:
                # Use the enhanced model loading method from Trainer
                success = self.trainer.load_model(model_path)
                if success:
                    # Load hyperparameters from model if available
                    try:
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        if 'config' in checkpoint:
                            # Update config with model's hyperparameters
                            model_config = checkpoint['config']
                            for key, value in model_config.items():
                                if hasattr(self.config, key):
                                    setattr(self.config, key, value)
                        
                        # Update GUI with loaded hyperparameters
                        self.update_gui_from_config()
                        self.update_hyperparameter_display()
                        
                        print("Model hyperparameters loaded and applied")
                    except Exception as e:
                        print(f"Warning: Could not load hyperparameters from model: {e}")
                    
                    self.status_text.set("Model and hyperparameters loaded successfully.")
                    return True
                else:
                    messagebox.showwarning("Model Load Error", f"Could not load model weights from {model_path}.\nUsing a randomly initialized model.")
            except Exception as e:
                messagebox.showwarning("Model Load Error", f"Could not load model weights from {model_path}:\n{e}\n\nUsing a randomly initialized model.")
        else:
            messagebox.showwarning("Model Not Found", f"Model file '{model_path}' not found.\nThe model will have random weights and produce gibberish.")
        return False

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

    def apply_current_settings(self):
        """Apply current model settings to active training session"""
        try:
            # Update config from GUI values
            self.config.learning_rate = float(self.lr_var.get())
            self.config.weight_decay = float(self.wd_var.get())
            self.config.dropout = float(self.dropout_var.get())
            self.config.batch_size = int(self.batch_size_var.get())
            self.config.block_size = int(self.block_size_var.get())
            
            # Architecture parameters
            self.config.n_embd = int(self.embd_dim_var.get())
            self.config.n_layer = int(self.layers_var.get())
            self.config.n_head = int(self.heads_var.get())
            
            # MoE settings
            self.config.use_moe = self.use_moe_var.get()
            self.config.moe_num_experts = int(self.moe_experts_var.get())
            self.config.moe_k = int(self.moe_k_var.get())
            
            # Attention settings
            self.config.attention_type = self.attention_type_var.get()
            self.config.n_query_groups = int(self.query_groups_var.get())
            
            # Recreate trainer with new config
            if hasattr(self, 'trainer') and self.trainer:
                try:
                    old_trainer = self.trainer
                    self.log_to_console("Recreating trainer with new settings...", "INFO")
                    
                    # Clear GPU cache before creating new trainer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Create new trainer with updated config
                    self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
                    
                    # Try to transfer model weights if compatible
                    if hasattr(old_trainer, 'model') and hasattr(self.trainer, 'model'):
                        try:
                            old_state = old_trainer.model.state_dict()
                            new_state = self.trainer.model.state_dict()
                            
                            # Transfer compatible weights
                            compatible_keys = []
                            for key in old_state:
                                if key in new_state and old_state[key].shape == new_state[key].shape:
                                    new_state[key] = old_state[key]
                                    compatible_keys.append(key)
                            
                            if compatible_keys:
                                self.trainer.model.load_state_dict(new_state)
                                self.log_to_console(f"Transferred {len(compatible_keys)} compatible layers", "SUCCESS")
                            else:
                                self.log_to_console("No compatible weights to transfer - using fresh model", "WARNING")
                        except Exception as e:
                            self.log_to_console(f"Could not transfer weights: {e}", "WARNING")
                    
                    self.log_to_console("Applied current settings to training session", "SUCCESS")
                    
                except Exception as e:
                    self.log_to_console(f"Error recreating trainer: {e}", "ERROR")
                    # Fall back to just updating the config
                    if hasattr(self, 'trainer') and self.trainer:
                        self.trainer.config = self.config
                        self.log_to_console("Updated trainer config (fallback)", "WARNING")
            else:
                # Create new trainer if none exists
                try:
                    self.trainer = Trainer(self.config, loss_callback=self.queue_loss, metrics_callback=self.queue_metrics)
                    self.log_to_console("Created new trainer with current settings", "SUCCESS")
                except Exception as e:
                    self.log_to_console(f"Error creating trainer: {e}", "ERROR")
                
            # Update GUI fields to match applied values
            self.update_gui_from_config()
            
            # Update hyperparameter display
            if hasattr(self, 'update_hyperparameter_display'):
                self.update_hyperparameter_display()
            
            self.status_text.set("Settings applied successfully")
            
        except Exception as e:
            error_msg = f"Error applying settings: {e}"
            self.log_to_console(error_msg, "ERROR")
            messagebox.showerror("Apply Settings Error", error_msg)

    def reset_model_settings(self):
        """Reset model settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all model settings to defaults?"):
            try:
                # Reset to default config values
                from ShitGPT import Config
                default_config = Config()
                
                # Update GUI variables
                self.lr_var.set(str(default_config.learning_rate))
                self.wd_var.set(str(default_config.weight_decay))
                self.dropout_var.set(str(default_config.dropout))
                self.batch_size_var.set(str(default_config.batch_size))
                self.block_size_var.set(str(default_config.block_size))
                
                # Architecture
                self.embd_dim_var.set(str(default_config.n_embd))
                self.layers_var.set(str(default_config.n_layer))
                self.heads_var.set(str(default_config.n_head))
                
                # MoE
                self.use_moe_var.set(default_config.use_moe)
                self.moe_experts_var.set(str(default_config.moe_num_experts))
                self.moe_k_var.set(str(default_config.moe_k))
                
                # Attention
                self.attention_type_var.set(default_config.attention_type)
                self.query_groups_var.set(str(default_config.n_query_groups))
                
                # Reset VRAM profile to low
                self.vram_profile_var.set("low")
                self.on_vram_profile_change()
                
                self.log_to_console("Reset all settings to defaults", "INFO")
                self.status_text.set("Settings reset to defaults")
                
            except Exception as e:
                error_msg = f"Error resetting settings: {e}"
                self.log_to_console(error_msg, "ERROR")
                messagebox.showerror("Reset Error", error_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = GPT_GUI(root)
    root.mainloop()
