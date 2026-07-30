import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pheno_core

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class PhenoApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("PhenoPlot")
        self.geometry("1350x880")

        # Set Window and Taskbar Icon
        logo_path = os.path.join(os.path.dirname(__file__), "app_logo.png")
        if os.path.exists(logo_path):
            try:
                self.app_icon = tk.PhotoImage(file=logo_path)
                self.wm_iconphoto(True, self.app_icon)
            except Exception as e:
                print(f"Could not load window icon: {e}")

        # Global Bind Ctrl+A / Cmd+A for selecting all text
        self.bind_all("<Control-a>", self.select_all_text)
        self.bind_all("<Command-a>", self.select_all_text)

        self.db_filepath = "phenotypes.ppc"
        self.phenotype_coords = {}
        self.user_coords = {}

        # Decrypt master .ppc into memory (searches both local dir & PyInstaller bundle)
        self.phenotype_coords = pheno_core.load_phenotypes(self.db_filepath)

        self.tabview = ctk.CTkTabview(self, width=1320, height=840)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tab_plot = self.tabview.add("PCA Map")
        self.tab_calc = self.tabview.add("Phenotype Calculator")
        self.tab_data = self.tabview.add("Coordinate file")
        self.tab_midpoint = self.tabview.add("Midpoint simulator")

        self.build_plot_tab()
        self.build_calc_tab()
        self.build_data_tab()
        self.build_midpoint_tab()

        lbl_signature = ctk.CTkLabel(
            self,
            text="By Chuggy",
            font=("Helvetica", 11, "italic"),
            text_color="#666666",
        )
        lbl_signature.place(relx=0.98, rely=0.98, anchor="se")

    def select_all_text(self, event):
        widget = event.widget
        if hasattr(widget, "select_range"):
            widget.select_range(0, "end")
            return "break"
        elif hasattr(widget, "tag_add"):
            widget.tag_add("sel", "1.0", "end")
            return "break"

    # ------------------------------------------------------------------
    # 1. PCA MAP TAB
    # ------------------------------------------------------------------
    def build_plot_tab(self):
        control_panel = ctk.CTkFrame(self.tab_plot, width=220)
        control_panel.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(
            control_panel,
            text="PhenoPlot PCA",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=10)

        btn_refresh = ctk.CTkButton(
            control_panel, text="Refresh Map", command=self.update_plot
        )
        btn_refresh.pack(pady=10, padx=15, fill="x")

        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=120, facecolor="#121212")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(
            side="right", fill="both", expand=True, padx=10, pady=10
        )

        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#1b1b1b")
        self.fig.patch.set_facecolor("#121212")

        valid_builtin = {
            k: v for k, v in self.phenotype_coords.items() if len(v) == 13
        }
        valid_user = {
            k: v for k, v in self.user_coords.items() if len(v) == 13
        }

        if not valid_builtin:
            self.canvas.draw_idle()
            return

        pheno_names = list(valid_builtin.keys())
        raw_matrix = [
            pheno_core.transform_features(valid_builtin[name])
            for name in pheno_names
        ]
        data_matrix = np.array(raw_matrix)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        weighted_data = scaled_data * pheno_core.TRAIT_WEIGHTS

        pca = PCA(n_components=2)
        pheno_coords_2d = pca.fit_transform(weighted_data)

        # Plot Reference Phenotypes in Fixed PCA Space
        self.ax.scatter(
            pheno_coords_2d[:, 0],
            pheno_coords_2d[:, 1],
            color="#00f0ff",
            s=35,
            edgecolors="#ffffff",
            linewidths=0.4,
            alpha=0.85,
            zorder=3,
        )

        for i, name in enumerate(pheno_names):
            angle = (i * 47) % 360
            rad = np.radians(angle)
            dx = 8 * np.cos(rad)
            dy = 8 * np.sin(rad)

            self.ax.annotate(
                name,
                (pheno_coords_2d[i, 0], pheno_coords_2d[i, 1]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=5.5,
                fontweight="bold",
                alpha=0.9,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="#181818",
                    ec="#333333",
                    lw=0.4,
                    alpha=0.5,
                ),
            )

        # Plot Custom User Coordinates via Top-5 Power-4 Weighted Interpolation
        for user_name, user_raw_vec in valid_user.items():
            user_features = pheno_core.transform_features(user_raw_vec)
            scaled_user = scaler.transform([user_features])[0]
            weighted_user = scaled_user * pheno_core.TRAIT_WEIGHTS

            distances_13d = np.linalg.norm(weighted_data - weighted_user, axis=1)
            max_dist = np.max(distances_13d) if np.max(distances_13d) > 0 else 1.0
            similarities = np.exp(-distances_13d / (max_dist * 0.35)) * 100

            results = sorted(
                zip(pheno_names, similarities, pheno_coords_2d),
                key=lambda x: x[1],
                reverse=True,
            )
            top_5_matches = results[:5]

            raw_pcts = np.array([pct for _, pct, _ in top_5_matches])
            weighted_pcts = raw_pcts**4
            sum_pcts = np.sum(weighted_pcts)
            normalized_weights = (
                weighted_pcts / sum_pcts if sum_pcts > 0 else np.ones(5) / 5.0
            )

            top_2d_coords = np.array([coords for _, _, coords in top_5_matches])
            user_2d = np.sum(
                top_2d_coords * normalized_weights[:, np.newaxis], axis=0
            )

            self.ax.scatter(
                user_2d[0],
                user_2d[1],
                color="#ff0055",
                s=110,
                marker="*",
                edgecolors="#ffffff",
                linewidths=0.7,
                zorder=5,
            )
            self.ax.annotate(
                user_name,
                (user_2d[0], user_2d[1]),
                color="#ff0055",
                fontsize=7.5,
                fontweight="bold",
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
            )

        self.ax.grid(True, linestyle=":", alpha=0.15, color="#ffffff")
        self.ax.axhline(0, color="#444444", linestyle="-", linewidth=0.8, alpha=0.4)
        self.ax.axvline(0, color="#444444", linestyle="-", linewidth=0.8, alpha=0.4)

        self.ax.set_title(
            "GLOBAL PHENOTYPE PCA SPACE MAP",
            fontsize=11,
            fontweight="bold",
            color="#00f0ff",
            loc="left",
        )
        self.ax.set_xlabel(
            "Principal Component 1 (Primary Morphological Variance)",
            color="#888888",
            fontsize=7.5,
        )
        self.ax.set_ylabel(
            "Principal Component 2 (Cranial Metrics & Pigmentation)",
            color="#888888",
            fontsize=7.5,
        )
        self.ax.tick_params(colors="#888888", labelsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 2. PHENOTYPE CALCULATOR TAB
    # ------------------------------------------------------------------
    def build_calc_tab(self):
        frame = ctk.CTkFrame(self.tab_calc)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame, text="Phenotype Calculator Engine", font=("Helvetica", 18, "bold")
        ).pack(pady=10)

        top_bar = ctk.CTkFrame(frame)
        top_bar.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(top_bar, text="Select Target Profile:").pack(
            side="left", padx=5
        )
        self.combo_profiles = ctk.CTkComboBox(
            top_bar, values=["No profiles available"], width=220
        )
        self.combo_profiles.pack(side="left", padx=5)

        ctk.CTkLabel(top_bar, text="OR Raw Coordinate:").pack(
            side="left", padx=(15, 5)
        )
        self.entry_calc_raw = ctk.CTkEntry(
            top_bar, width=320, placeholder_text='"Sample": [0.4512, 0.1241, ...]'
        )
        self.entry_calc_raw.pack(side="left", padx=5)

        btn_run = ctk.CTkButton(
            top_bar, text="Calculate Best Matches", command=self.run_calculator
        )
        btn_run.pack(side="left", padx=10)

        self.txt_calc_results = ctk.CTkTextbox(
            frame, height=450, font=("Courier", 13)
        )
        self.txt_calc_results.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_calc_results.configure(state="disabled")

        self.refresh_calculator_dropdown()

    def refresh_calculator_dropdown(self):
        names = list(self.user_coords.keys()) + list(self.phenotype_coords.keys())
        if names:
            self.combo_profiles.configure(values=names)
            self.combo_profiles.set(names[0])

    def run_calculator(self):
        raw_str = self.entry_calc_raw.get().strip()
        target_name = self.combo_profiles.get()
        all_coords = {**self.user_coords, **self.phenotype_coords}

        target_vec = None
        display_name = ""

        if raw_str:
            try:
                display_name, target_vec = pheno_core.parse_coordinate_with_name(raw_str)
            except Exception as e:
                self.txt_calc_results.configure(state="normal")
                self.txt_calc_results.delete("1.0", "end")
                self.txt_calc_results.insert("1.0", f"Error parsing input coordinates:\n{e}")
                self.txt_calc_results.configure(state="disabled")
                return
        elif target_name in all_coords:
            target_vec = all_coords[target_name]
            display_name = target_name
        else:
            return

        pheno_names = list(self.phenotype_coords.keys())
        raw_matrix = [
            pheno_core.transform_features(self.phenotype_coords[name])
            for name in pheno_names
        ]
        data_matrix = np.array(raw_matrix)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        weighted_data = scaled_data * pheno_core.TRAIT_WEIGHTS

        user_features = pheno_core.transform_features(target_vec)
        scaled_user = scaler.transform([user_features])[0]
        weighted_user = scaled_user * pheno_core.TRAIT_WEIGHTS

        distances_13d = np.linalg.norm(weighted_data - weighted_user, axis=1)
        max_dist = np.max(distances_13d) if np.max(distances_13d) > 0 else 1.0
        similarities = np.exp(-distances_13d / (max_dist * 0.35)) * 100

        results = sorted(
            zip(pheno_names, similarities, distances_13d),
            key=lambda x: x[1],
            reverse=True,
        )

        self.txt_calc_results.configure(state="normal")
        self.txt_calc_results.delete("1.0", "end")
        self.txt_calc_results.insert("end", f"====================================================\n")
        self.txt_calc_results.insert("end", f"   PHENOTYPE MATCH RESULTS FOR: {display_name}\n")
        self.txt_calc_results.insert("end", f"====================================================\n\n")
        self.txt_calc_results.insert("end", f"{'Rank':<6}{'Phenotype':<25}{'Similarity':<15}{'13D Distance':<10}\n")
        self.txt_calc_results.insert("end", f"----------------------------------------------------\n")

        for idx, (p_name, sim, dist) in enumerate(results[:15], 1):
            self.txt_calc_results.insert(
                "end", f"{idx:<6}{p_name:<25}{sim:.1f}%{'':<10}{dist:.2f}\n"
            )

        self.txt_calc_results.configure(state="disabled")

    # ------------------------------------------------------------------
    # 3. COORDINATE FILE TAB
    # ------------------------------------------------------------------
    def build_data_tab(self):
        frame = ctk.CTkFrame(self.tab_data)
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Top Control Bar
        top_bar = ctk.CTkFrame(frame, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=(5, 10))

        btn_load_ppc = ctk.CTkButton(
            top_bar,
            text="Load .ppc file",
            command=self.load_custom_ppc_file,
            width=150
        )
        btn_load_ppc.pack(side="left", padx=5)

        btn_reload_ppc = ctk.CTkButton(
            top_bar,
            text="Reload phenotypes.ppc",
            command=self.reload_default_ppc,
            width=160,
            fg_color="#3a3a3a",
            hover_color="#4a4a4a"
        )
        btn_reload_ppc.pack(side="left", padx=5)

        btn_patreon = ctk.CTkButton(
            top_bar,
            text="❤️ Get your custom coordinates here",
            command=self.open_patreon,
            fg_color="#FF424D",
            hover_color="#D9363E",
            font=("Helvetica", 12, "bold"),
            width=250
        )
        btn_patreon.pack(side="right", padx=5)

        columns_frame = ctk.CTkFrame(frame)
        columns_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Source Panel
        col_source = ctk.CTkFrame(columns_frame)
        col_source.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(
            col_source,
            text="Source (Reference Phenotypes)",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=5)
        self.txt_source = ctk.CTkTextbox(col_source, height=350, font=("Courier", 11))
        self.txt_source.pack(fill="both", expand=True, padx=5, pady=5)

        # Target Panel
        col_target = ctk.CTkFrame(columns_frame)
        col_target.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(
            col_target,
            text="Target (User / Custom Coordinates)",
            font=("Helvetica", 14, "bold"),
            text_color="#ff0055",
        ).pack(pady=5)
        self.txt_target = ctk.CTkTextbox(col_target, height=350, font=("Courier", 11))
        self.txt_target.pack(fill="both", expand=True, padx=5, pady=5)

        btn_apply = ctk.CTkButton(
            frame,
            text="Apply Target Coordinates to PCA Map & Calculator",
            command=self.apply_target_coordinates,
        )
        btn_apply.pack(pady=10)

        self.update_data_textboxes()

    def load_custom_ppc_file(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Custom Phenotype File(s)",
            filetypes=[("PhenoPlot Collection", "*.ppc"), ("All Files", "*.*")]
        )
        if file_paths:
            for fpath in file_paths:
                loaded_data = pheno_core.load_phenotypes(fpath)
                if loaded_data:
                    self.phenotype_coords.update(loaded_data)
            
            self.update_data_textboxes()
            self.refresh_calculator_dropdown()
            self.update_plot()

    def reload_default_ppc(self):
        self.db_filepath = "phenotypes.ppc"
        self.phenotype_coords = pheno_core.load_phenotypes(self.db_filepath)
        self.user_coords.clear()
        self.update_data_textboxes()
        self.refresh_calculator_dropdown()
        self.update_plot()

    def open_patreon(self):
        webbrowser.open("https://www.patreon.com/cw/PhenoPlot")

    def update_data_textboxes(self):
        self.txt_source.delete("1.0", "end")
        for name, vec in self.phenotype_coords.items():
            formatted_line = pheno_core.format_coordinate(name, vec)
            self.txt_source.insert("end", f"{formatted_line}\n")

        self.txt_target.delete("1.0", "end")
        for name, vec in self.user_coords.items():
            formatted_line = pheno_core.format_coordinate(name, vec)
            self.txt_target.insert("end", f"{formatted_line}\n")

    def apply_target_coordinates(self):
        text = self.txt_target.get("1.0", "end").strip()
        lines = text.split("\n")
        self.user_coords = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].replace('"', "").replace("'", "").strip()
                try:
                    coords = pheno_core.parse_coordinate_string(parts[1])
                    self.user_coords[name] = coords
                except Exception as e:
                    print(f"Error parsing coordinate '{line}': {e}")

        self.refresh_calculator_dropdown()
        self.update_plot()
        self.tabview.set("PCA Map")

    # ------------------------------------------------------------------
    # 4. MIDPOINT SIMULATOR TAB
    # ------------------------------------------------------------------
    def build_midpoint_tab(self):
        frame = ctk.CTkFrame(self.tab_midpoint)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame, text="Midpoint simulator", font=("Helvetica", 18, "bold")
        ).pack(pady=10)

        ctk.CTkLabel(
            frame, text="Set A (Name or Coordinate):"
        ).pack(anchor="w", padx=20, pady=(10, 2))
        self.entry_mid_a = ctk.CTkEntry(
            frame,
            width=750,
            placeholder_text='e.g. Hallstatt OR "SampleA": [0.4512, 0.1241, ...]',
        )
        self.entry_mid_a.pack(anchor="w", padx=20, pady=5)

        ctk.CTkLabel(
            frame, text="Set B (Name or Coordinate):"
        ).pack(anchor="w", padx=20, pady=(10, 2))
        self.entry_mid_b = ctk.CTkEntry(
            frame,
            width=750,
            placeholder_text='e.g. Borreby OR "SampleB": [0.4512, 0.1241, ...]',
        )
        self.entry_mid_b.pack(anchor="w", padx=20, pady=5)

        ctk.CTkLabel(frame, text="Simulated Coordinate Name:").pack(
            anchor="w", padx=20, pady=(10, 2)
        )
        self.entry_mid_name = ctk.CTkEntry(
            frame, width=300, placeholder_text="e.g. Simulated_Midpoint"
        )
        self.entry_mid_name.pack(anchor="w", padx=20, pady=5)

        btn_sim = ctk.CTkButton(
            frame, text="Calculate Midpoint & Plot", command=self.calculate_midpoint
        )
        btn_sim.pack(pady=25)

    def calculate_midpoint(self):
        input_a = self.entry_mid_a.get().strip()
        input_b = self.entry_mid_b.get().strip()
        out_name = self.entry_mid_name.get().strip() or "Simulated_Midpoint"

        all_known = {**self.phenotype_coords, **self.user_coords}

        try:
            if input_a in all_known:
                vec_a = all_known[input_a]
            else:
                vec_a = pheno_core.parse_coordinate_string(input_a)

            if input_b in all_known:
                vec_b = all_known[input_b]
            else:
                vec_b = pheno_core.parse_coordinate_string(input_b)
        except Exception as e:
            print(f"Error parsing midpoint inputs: {e}")
            return

        mid_vec = pheno_core.calculate_coordinate_midpoint(vec_a, vec_b)
        self.user_coords[out_name] = mid_vec

        self.update_data_textboxes()
        self.refresh_calculator_dropdown()
        self.update_plot()
        self.tabview.set("PCA Map")


if __name__ == "__main__":
    app = PhenoApp()
    app.mainloop()
