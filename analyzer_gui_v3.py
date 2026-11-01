"""
NanoCyte - GUI Application 

Select both grayscale DAPI and nanoparticle (NP) TIFF images. Images are
paired by a shared filename stem, for example:
    field1_dapi.tif
    field1_np.tif

The DAPI image is used only for nucleus segmentation; the NP image is used
only for fluorescence quantification.
"""

import os
import platform
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import pandas as pd

from nanocyte import CellFluorescenceAnalyzer


class CellFluorescenceGUI:
    """GUI for DAPI-segmented nanoparticle fluorescence analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("NanoCyte - Cell Analysis Tool")
        self.root.geometry("1050x850")
        self.root.minsize(940, 720)

        self.image_files = []
        self.output_folder = None
        self.analyzer = None
        self.stop_flag = False
        self.analysis_thread = None
        self.log_queue = queue.Queue()

        self.setup_ui()
        self.check_log_queue()

    # User interface
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        ttk.Label(
            main_frame,
            text="NanoCyte Fluorescence Analyzer",
            font=("Arial", 18, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=8)

        ttk.Label(
            main_frame,
            text=(
                "Select grayscale DAPI and NP TIFFs together. Files must be paired "
                "by name, e.g. field1_dapi.tif + field1_np.tif."
            ),
            wraplength=950,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="1. Select Images", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(
            file_frame, text="Browse Files...", command=self.browse_files
        ).grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)

        self.file_count_label = ttk.Label(file_frame, text="No files selected")
        self.file_count_label.grid(row=0, column=1, sticky=tk.W, padx=10)

        ttk.Button(
            file_frame, text="Select Output Folder...", command=self.select_output_folder
        ).grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)

        self.output_label = ttk.Label(file_frame, text="Output: Same as input folder")
        self.output_label.grid(row=1, column=1, sticky=tk.W, padx=10)

        self.selected_files_label = ttk.Label(
            file_frame, text="Expected suffixes: _dapi and _np", wraplength=850
        )
        self.selected_files_label.grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(5, 0)
        )

        # Parameters
        param_frame = ttk.LabelFrame(
            main_frame, text="2. Segmentation and Measurement Parameters", padding="10"
        )
        param_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        for col in range(4):
            param_frame.columnconfigure(col, weight=0)
        param_frame.columnconfigure(3, weight=1)

        row = 0
        ttk.Label(param_frame, text="Minimum Nucleus Area:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.min_nucleus_var = tk.IntVar(value=150)
        ttk.Spinbox(
            param_frame, from_=1, to=100000, textvariable=self.min_nucleus_var, width=10
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(param_frame, text="pixels").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="Maximum Nucleus Area:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.max_nucleus_var = tk.IntVar(value=5000)
        ttk.Spinbox(
            param_frame, from_=10, to=1000000, textvariable=self.max_nucleus_var, width=10
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(param_frame, text="pixels").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="Minimum Circularity:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.circularity_var = tk.DoubleVar(value=0.25)
        self.circularity_slider = ttk.Scale(
            param_frame,
            from_=0.0,
            to=1.0,
            variable=self.circularity_var,
            orient=tk.HORIZONTAL,
            length=180,
            command=self.update_circularity_label,
        )
        self.circularity_slider.grid(row=row, column=1, padx=5, sticky=tk.W)
        self.circularity_value_label = ttk.Label(param_frame, text="0.25", width=6)
        self.circularity_value_label.grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="Measurement Radius:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.cell_var = tk.IntVar(value=20)
        ttk.Spinbox(
            param_frame, from_=0, to=200, textvariable=self.cell_var, width=10
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(param_frame, text="pixels outside nucleus").grid(
            row=row, column=2, columnspan=2, sticky=tk.W
        )

        row += 1
        ttk.Label(param_frame, text="NP Background Sigma:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.background_sigma_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(
            param_frame,
            from_=0.5,
            to=100,
            increment=0.5,
            textvariable=self.background_sigma_var,
            width=10,
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(param_frame, text="pixels").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="DAPI Background Sigma:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.dapi_sigma_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.dapi_sigma_var,
            width=10,
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(param_frame, text="pixels").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="DAPI Threshold Factor:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.threshold_factor_var = tk.DoubleVar(value=0.050)
        ttk.Spinbox(
            param_frame,
            from_=0.05,
            to=1.5,
            increment=0.05,
            textvariable=self.threshold_factor_var,
            width=10,
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(
            param_frame, text="lower = more permissive; inspect segmentation"
        ).grid(row=row, column=2, columnspan=2, sticky=tk.W)

        row += 1
        ttk.Label(param_frame, text="Watershed Marker Distance:").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        self.watershed_distance_var = tk.IntVar(value=5)
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=50,
            textvariable=self.watershed_distance_var,
            width=10,
        ).grid(row=row, column=1, padx=5, sticky=tk.W)
        ttk.Label(
            param_frame, text="pixels; smaller may split closer nuclei"
        ).grid(row=row, column=2, columnspan=2, sticky=tk.W)
        
        # Actions
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=2, pady=8)

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))

        self.analyze_button = ttk.Button(
            action_frame, text="Start Analysis", command=self.start_analysis,
            style="Accent.TButton"
        )
        self.analyze_button.pack(side=tk.LEFT, padx=8)

        self.stop_button = ttk.Button(
            action_frame, text="Stop", command=self.stop_analysis, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)

        ttk.Button(
            action_frame, text="View Results Folder", command=self.open_results_folder
        ).pack(side=tk.LEFT, padx=8)

        # Progress/log
        log_frame = ttk.LabelFrame(main_frame, text="Progress & Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            log_frame, variable=self.progress_var, mode="determinate"
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, wrap=tk.WORD
        )
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("success", foreground="green")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        ).grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

    def update_circularity_label(self, value):
        self.circularity_value_label.config(text=f"{float(value):.2f}")

    ##########
    # Logging
    ##########
    def log(self, message, level="INFO"):
        self.log_queue.put((str(message), level))

    def check_log_queue(self):
        try:
            while True:
                message, level = self.log_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                start = self.log_text.index(tk.END)
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                end = self.log_text.index(tk.END)
                if level == "ERROR":
                    self.log_text.tag_add("error", start, end)
                elif level == "SUCCESS":
                    self.log_text.tag_add("success", start, end)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.check_log_queue)

    # File selection / pairing
    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select matched DAPI and NP grayscale TIFF images",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not files:
            return

        self.image_files = [Path(p) for p in files]
        pairs, errors = self.pair_selected_files(self.image_files)

        self.file_count_label.config(
            text=f"{len(self.image_files)} file(s) selected; {len(pairs)} pair(s) found"
        )
        preview = ", ".join(f"{d.name} + {n.name}" for d, n in pairs[:3])
        if len(pairs) > 3:
            preview += f", ... (+{len(pairs) - 3} more pairs)"
        if errors:
            preview += (" | " if preview else "") + f"{len(errors)} unmatched file(s)"
        self.selected_files_label.config(text=preview or "No valid pairs detected.")
        self.log(f"Selected {len(self.image_files)} file(s); found {len(pairs)} matched pair(s).")
        for error in errors:
            self.log(error, "ERROR")

        if self.output_folder is None and self.image_files:
            self.output_folder = self.image_files[0].parent
            self.output_label.config(text=f"Output: {self.output_folder}")

    @staticmethod
    def pair_selected_files(paths):
        """Pair files with _dapi/-dapi and _np/-np suffixes, case-insensitively."""
        dapi_by_stem = {}
        np_by_stem = {}
        errors = []

        for path in paths:
            p = Path(path)
            suffix = p.suffix.lower()
            if suffix not in (".tif", ".tiff"):
                errors.append(f"Skipping non-TIFF file: {p.name}")
                continue

            stem = p.stem
            lower = stem.lower()
            if lower.endswith("_dapi") or lower.endswith("-dapi"):
                key = lower[:-5]
                if key in dapi_by_stem:
                    errors.append(f"Duplicate DAPI image for pair '{key}': {p.name}")
                else:
                    dapi_by_stem[key] = p
            elif lower.endswith("_np") or lower.endswith("-np"):
                key = lower[:-3]
                if key in np_by_stem:
                    errors.append(f"Duplicate NP image for pair '{key}': {p.name}")
                else:
                    np_by_stem[key] = p
            else:
                errors.append(
                    f"Could not identify channel suffix for {p.name}; expected _dapi or _np."
                )

        pairs = []
        for key in sorted(set(dapi_by_stem) & set(np_by_stem)):
            pairs.append((dapi_by_stem[key], np_by_stem[key]))

        for key in sorted(set(dapi_by_stem) - set(np_by_stem)):
            errors.append(f"No NP partner selected for DAPI image: {dapi_by_stem[key].name}")
        for key in sorted(set(np_by_stem) - set(dapi_by_stem)):
            errors.append(f"No DAPI partner selected for NP image: {np_by_stem[key].name}")

        return pairs, errors

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self.output_folder = Path(folder)
            self.output_label.config(text=f"Output: {self.output_folder}")
            self.log(f"Output folder: {self.output_folder}", "SUCCESS")


    # Analysis
    def start_analysis(self):
        if not self.image_files:
            messagebox.showwarning(
                "No Files", "Select both DAPI and NP TIFFs for at least one field."
            )
            return

        pairs, errors = self.pair_selected_files(self.image_files)
        if errors:
            details = "\n".join(errors[:12])
            if len(errors) > 12:
                details += f"\n... and {len(errors) - 12} more issue(s)."
            if not pairs:
                messagebox.showerror("Image Pairing Problem", details)
                return
            proceed = messagebox.askyesno(
                "Image Pairing Warnings",
                f"{details}\n\nContinue with the {len(pairs)} valid pair(s)?",
            )
            if not proceed:
                return
        if not pairs:
            messagebox.showerror(
                "No Matched Pairs",
                "Select files named like field1_dapi.tif and field1_np.tif.",
            )
            return

        try:
            params = self.read_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self.analyze_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_var.set("Starting analysis...")
        self.stop_flag = False

        self.analysis_thread = threading.Thread(
            target=self.run_analysis, args=(pairs, params), daemon=True
        )
        self.analysis_thread.start()
        
    def stop_analysis(self):
        """Request a graceful stop after the current image pair."""
        if self.analysis_thread is None or not self.analysis_thread.is_alive():
            self.log("No analysis is currently running.")
            return

        self.stop_flag = True
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopping after the current image pair...")
        self.log(
            "Stop requested. Finishing the current image pair before stopping.",
            "WARNING"
        )
        
    def read_parameters(self):
        params = {
            "min_nucleus_area": int(self.min_nucleus_var.get()),
            "max_nucleus_area": int(self.max_nucleus_var.get()),
            "min_circularity": float(self.circularity_var.get()),
            "cell_radius": int(self.cell_var.get()),
            "background_sigma": float(self.background_sigma_var.get()),
            "dapi_background_sigma": float(self.dapi_sigma_var.get()),
            "threshold_factor": float(self.threshold_factor_var.get()),
            #"min_object_size": int(self.min_object_var.get()),
            "watershed_min_distance": int(self.watershed_distance_var.get()),
        }
        if params["min_nucleus_area"] < 1:
            raise ValueError("Minimum nucleus area must be at least 1.")
        if params["max_nucleus_area"] < params["min_nucleus_area"]:
            raise ValueError("Maximum nucleus area must be >= minimum nucleus area.")
        if params["cell_radius"] < 0:
            raise ValueError("Measurement radius must be >= 0.")
        if params["background_sigma"] <= 0 or params["dapi_background_sigma"] <= 0:
            raise ValueError("Both background sigma values must be > 0.")
        if params["threshold_factor"] <= 0:
            raise ValueError("DAPI threshold factor must be > 0.")
        return params

    def run_analysis(self, pairs, params):
        all_results = []
        output_dir = Path(self.output_folder) if self.output_folder else pairs[0][0].parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log("=" * 60)
        self.log("STARTING NANOPARTICLE FLUORESCENCE ANALYSIS", "SUCCESS")
        self.log(f"Parameters: {params}")
        self.log(f"Matched image pairs: {len(pairs)}")
        self.log("")

        try:
            for idx, (dapi_path, np_path) in enumerate(pairs):
                if self.stop_flag:
                    self.log("Analysis stopped by user.")
                    break

                self.root.after(
                    0,
                    lambda name=dapi_path.stem: self.status_var.set(f"Processing: {name}"),
                )
                self.root.after(
                    0, lambda p=(idx / len(pairs)) * 100: self.progress_var.set(p)
                )
                self.log(f"Processing ({idx + 1}/{len(pairs)}): {dapi_path.name} + {np_path.name}")

                try:
                    analyzer = CellFluorescenceAnalyzer(**params)
                    self.analyzer = analyzer
                    results = analyzer.analyze(
                        dapi_path, np_path, output_dir=output_dir, save_outputs=True
                    )

                    results = results.copy()
                    results["image_file"] = dapi_path.name
                    results["dapi_file"] = dapi_path.name
                    results["np_file"] = np_path.name
                    all_results.append(results)

                    n_cells = len(results)
                    if n_cells:
                        mean_value = results["mean_cell_fluorescence"].mean()
                        # median_value = results["median_cell_fluorescence"].median()
                        self.log(
                            f"  Quantified {n_cells} cells "
                            f"(mean fluorescence: {mean_value:.3f}; "
                            # f"median: {median_value:.3f})",
                            "SUCCESS",
                        )
                    else:
                        self.log(
                            "  No nuclei passed the current segmentation criteria. "
                            "Inspect the DAPI segmentation and consider adjusting settings.",
                            "ERROR",
                        )
                    self.log("")
                except Exception as exc:
                    self.log(
                        f"  Error processing pair {dapi_path.name} + {np_path.name}: {exc}",
                        "ERROR",
                    )
                    self.log("")

                self.root.after(
                    0, lambda p=((idx + 1) / len(pairs)) * 100: self.progress_var.set(p)
                )

            if all_results:
                combined = pd.concat(all_results, ignore_index=True)
                combined_path = output_dir / "all_images_cell_fluorescence.csv"
                combined.to_csv(combined_path, index=False)
                self.log(f"Saved combined results: {combined_path}", "SUCCESS")

            if self.stop_flag:
                self.root.after(0, lambda: self.status_var.set("Stopped"))
            else:
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                self.log("=" * 60, "SUCCESS")
                self.log("ANALYSIS COMPLETE", "SUCCESS")
                self.log("=" * 60, "SUCCESS")

        except Exception as exc:
            self.log(f"Fatal error: {exc}", "ERROR")
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

    # Open results folder
    def open_results_folder(self):
        folder = self.output_folder
        if folder is None and self.image_files:
            folder = self.image_files[0].parent
        if folder and Path(folder).exists():
            try:
                folder = str(folder)
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.call(["open", folder])
                else:
                    subprocess.call(["xdg-open", folder])
            except Exception as exc:
                messagebox.showerror("Could Not Open Folder", str(exc))
        else:
            messagebox.showinfo("No Results", "No output folder has been selected yet.")


def main():
    root = tk.Tk()
    CellFluorescenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
