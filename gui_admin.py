import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
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

        self.title("PhenoPlot - Admin")
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

        if os.path.exists(self.db_filepath):
            self.phenotype_coords = pheno_core.load_phenotypes(self.db_filepath)

        self.tabview = ctk.CTkTabview(self, width=1320, height=840)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tab_plot = self.tabview.add("PCA Map")
        self.tab_calc = self.tabview.add("Phenotype Calculator")
        self.tab_quest = self.tabview.add("Create coordinate")
        self.tab_data = self.tabview.add("Coordinate file")
        self.tab_midpoint = self.tabview.add("Midpoint simulator")

        self.build_plot_tab()
        self.build_calc_tab()
        self.build_quest_tab()
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
    # 1. PCA MAP TAB (EXACT DEMO PCA LOGIC)
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

        # Safeguard: Filter to 13-element vectors
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
    # 3. CREATE COORDINATE TAB
    # ------------------------------------------------------------------
    def build_quest_tab(self):
        scroll_frame = ctk.CTkScrollableFrame(self.tab_quest)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            scroll_frame, text="Create Coordinates", font=("Helvetica", 18, "bold")
        ).pack(pady=10)

        top_frame = ctk.CTkFrame(scroll_frame)
        top_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(top_frame, text="Coordinate Name:").pack(side="left", padx=10)
        self.entry_profile_name = ctk.CTkEntry(
            top_frame, width=220, placeholder_text="e.g. My_Coordinates"
        )
        self.entry_profile_name.pack(side="left", padx=10)

        self.var_gender = ctk.StringVar(value="Male")
        gender_seg = ctk.CTkSegmentedButton(
            top_frame,
            values=["Male", "Female"],
            variable=self.var_gender,
            command=self.on_gender_change,
        )
        gender_seg.pack(side="right", padx=10)

        self.sliders = {}

        self.create_category_header(scroll_frame, "Macro markers")
        self.add_marked_slider(
            scroll_frame,
            "Prognathism",
            [("Orthognathic", 1.0), ("Mesognathic", 2.2), ("Prognathic", 4.5)],
            is_precision_01=True,
        )
        self.add_marked_slider(
            scroll_frame,
            "Hairtype",
            [
                ("Coarse Straight", 1.0),
                ("Straight", 4.0),
                ("Wavy", 5.0),
                ("Curly", 6.0),
                ("Kinky", 9.5),
                ("Peppercorn", 12.0),
            ],
        )
        # Updated to is_precision_01=True for 0.1 step increments
        self.add_marked_slider(
            scroll_frame,
            "Eye Shape",
            [
                ("Standard Open", 1.0),
                ("Slight Fold", 2.2),
                ("Full Epicanthic Fold", 4.5),
            ],
            is_precision_01=True,
        )

        self.create_category_header(scroll_frame, "Skull/Face")
        self.add_marked_slider(
            scroll_frame,
            "FWHR",
            [
                ("Gracile Narrow", 1.6),
                ("Medium", 1.8),
                ("Cromagnid Broad", 2.1),
            ],
            is_precision_01=True,
        )
        self.add_marked_slider(
            scroll_frame,
            "Cephalic Index",
            [
                ("Hyperdolichocephalic", 67.5),
                ("Dolichocephalic", 72.5),
                ("Mesocephalic", 77.5),
                ("Brachycephalic", 82.5),
                ("Hyperbrachycephalic", 87.5),
            ],
        )
        self.add_marked_slider(
            scroll_frame,
            "Height-length Index",
            [
                ("Chamaecranic", 65.0),
                ("Orthocranic", 72.5),
                ("Hypsicranic", 80.0),
            ],
        )
        self.add_marked_slider(
            scroll_frame,
            "Nasal Index",
            [
                ("Hyperleptorrhine", 50.0),
                ("Leptorrhine", 60.0),
                ("Mesorrhine", 77.0),
                ("Platyrrhine", 92.0),
                ("Hyperplatyrrhine", 105.0),
            ],
        )
        self.add_marked_slider(
            scroll_frame,
            "Facial Index",
            [
                ("Hypereuryprosopic", 77.0),
                ("Euryprosopic", 81.5),
                ("Mesoprosopic", 86.0),
                ("Leptoprosopic", 91.0),
                ("Hyperleptoprosopic", 96.5),
            ],
            custom_range=(75.0, 100.0),
        )

        self.create_category_header(scroll_frame, "Pigment")
        self.add_marked_slider(
            scroll_frame,
            "Hair color",
            [
                ("1: Black", 1.0),
                ("2: Dark Brown", 2.0),
                ("3: Light Brown", 3.0),
                ("4: Red / Ginger", 4.0),
                ("5: Blonde", 5.0),
                ("6: Platinum Blonde", 6.0),
            ],
        )
        self.add_marked_slider(
            scroll_frame,
            "Skin color",
            [
                ("1: Very Pale", 1.0),
                ("2: Fair/Light", 2.0),
                ("3: Medium/Olive", 3.0),
                ("4: Dark Brown/Olive", 4.0),
                ("5: Dark Brown/Black", 5.0),
                ("6: Deeply Pigmented", 6.0),
            ],
        )
        self.add_marked_slider(
            scroll_frame,
            "Eye color",
            [
                ("1: Dark Brown", 1.0),
                ("2: Brown", 2.0),
                ("3: Hazel", 3.0),
                ("4: Green", 4.0),
                ("5: Blue", 5.0),
                ("6: Light Blue / Grey", 6.0),
            ],
        )

        self.create_category_header(scroll_frame, "Body")
        self.add_height_slider(scroll_frame)
        self.add_marked_slider(
            scroll_frame,
            "Somatotype",
            [("Ectomorph", 1.0), ("Mesomorph", 2.0), ("Endomorph", 3.0)],
        )

        action_frame = ctk.CTkFrame(scroll_frame)
        action_frame.pack(pady=25)

        btn_add_and_export = ctk.CTkButton(
            action_frame,
            text="Add to Map & Calculator (.ppc)",
            fg_color="#8a2be2",
            hover_color="#6a1b9a",
            command=self.add_and_export_coordinate,
        )
        btn_add_and_export.pack(padx=10)

    def create_category_header(self, parent, text):
        lbl = ctk.CTkLabel(
            parent,
            text=f"--- {text} ---",
            font=("Helvetica", 14, "bold"),
            text_color="#00e5ff",
        )
        lbl.pack(anchor="w", padx=20, pady=(20, 5))

    def add_marked_slider(
        self, parent, name, options, is_precision_01=False, custom_range=None
    ):
        f = ctk.CTkFrame(parent)
        f.pack(fill="x", padx=20, pady=12)

        min_v = custom_range[0] if custom_range else options[0][1]
        max_v = custom_range[1] if custom_range else options[-1][1]
        default_v = options[len(options) // 2][1]

        lbl_header = ctk.CTkLabel(
            f,
            text=f"{name}: {default_v:.2f}"
            if is_precision_01
            else f"{name}: {default_v}",
            font=("Helvetica", 13, "bold"),
            anchor="w",
        )
        lbl_header.pack(anchor="w", padx=15, pady=(8, 2))

        markers_frame = ctk.CTkFrame(f, fg_color="transparent")
        markers_frame.pack(fill="x", padx=15, pady=(2, 2))

        for text, val in options:
            btn_mark = ctk.CTkButton(
                markers_frame,
                text=f"{text}\n({val})",
                font=("Helvetica", 8),
                fg_color="#2b2b2b",
                hover_color="#3a3a3a",
                text_color="#cccccc",
                height=28,
                command=lambda v=val, n=name, prec=is_precision_01: self.set_slider_value(
                    n, v, prec
                ),
            )
            btn_mark.pack(side="left", expand=True, fill="x", padx=1)

        num_steps = (
            int((max_v - min_v) * 10)
            if is_precision_01
            else int((max_v - min_v) * 2)
        )
        slider = ctk.CTkSlider(
            f,
            from_=min_v,
            to=max_v,
            number_of_steps=num_steps,
            command=lambda val, l=lbl_header, n=name, prec=is_precision_01: self.update_slider_label(
                l, n, val, prec
            ),
        )
        slider.set(default_v)
        slider.pack(fill="x", padx=15, pady=(4, 10))

        self.sliders[name] = (slider, lbl_header, is_precision_01)

    def add_height_slider(self, parent):
        self.frame_height = ctk.CTkFrame(parent)
        self.frame_height.pack(fill="x", padx=20, pady=12)
        self.rebuild_height_ui()

    def rebuild_height_ui(self):
        for child in self.frame_height.winfo_children():
            child.destroy()

        is_female = self.var_gender.get() == "Female"

        if is_female:
            options = [
                ("Very Short", 150.0),
                ("Short", 155.0),
                ("Rather Short", 160.0),
                ("Medium", 165.0),
                ("Rather Tall", 170.0),
                ("Tall", 175.0),
                ("Very Tall", 180.0),
            ]
        else:
            options = [
                ("Very Short", 162.5),
                ("Short", 167.5),
                ("Rather Short", 172.5),
                ("Medium", 177.5),
                ("Rather Tall", 182.5),
                ("Tall", 187.5),
                ("Very Tall", 192.5),
            ]

        default_v = options[3][1]

        self.lbl_height = ctk.CTkLabel(
            self.frame_height,
            text=f"Height: {default_v:.1f} cm",
            font=("Helvetica", 13, "bold"),
            anchor="w",
        )
        self.lbl_height.pack(anchor="w", padx=15, pady=(8, 2))

        markers_frame = ctk.CTkFrame(self.frame_height, fg_color="transparent")
        markers_frame.pack(fill="x", padx=15, pady=(2, 2))

        for text, val in options:
            btn_mark = ctk.CTkButton(
                markers_frame,
                text=f"{text}\n({val})",
                font=("Helvetica", 8),
                fg_color="#2b2b2b",
                hover_color="#3a3a3a",
                text_color="#cccccc",
                height=28,
                command=lambda v=val: self.set_slider_value("Height", v, False),
            )
            btn_mark.pack(side="left", expand=True, fill="x", padx=1)

        slider_h = ctk.CTkSlider(
            self.frame_height,
            from_=100.0,
            to=220.0,
            number_of_steps=240,
            command=lambda val: self.update_slider_label(
                self.lbl_height, "Height", val, False
            ),
        )
        slider_h.set(default_v)
        slider_h.pack(fill="x", padx=15, pady=(4, 10))

        self.sliders["Height"] = (slider_h, self.lbl_height, False)

    def on_gender_change(self, choice):
        self.rebuild_height_ui()

    def update_slider_label(self, lbl, name, val, is_precision_01):
        final_v = (
            round(val, 2) if is_precision_01 else pheno_core.round_to_half(val)
        )
        unit = " cm" if name == "Height" else ""
        lbl.configure(text=f"{name}: {final_v}{unit}")

    def set_slider_value(self, name, val, is_precision_01):
        slider, lbl, _ = self.sliders[name]
        slider.set(val)
        self.update_slider_label(lbl, name, val, is_precision_01)

    def get_current_creator_vector(self):
        ordered_keys = [
            "Prognathism",
            "Hairtype",
            "Eye Shape",
            "FWHR",
            "Cephalic Index",
            "Height-length Index",
            "Nasal Index",
            "Facial Index",
            "Hair color",
            "Skin color",
            "Eye color",
            "Height",
            "Somatotype",
        ]
        raw_vec = []
        for k in ordered_keys:
            slider, _, is_precision_01 = self.sliders[k]
            v = slider.get()
            raw_vec.append(
                round(v, 2) if is_precision_01 else pheno_core.round_to_half(v)
            )

        is_female = self.var_gender.get() == "Female"
        return pheno_core.apply_gender_calibration(
            raw_vec, is_female=is_female
        )

    def add_and_export_coordinate(self):
        name = self.entry_profile_name.get().strip() or "Custom_Coordinate"
        vec = self.get_current_creator_vector()

        self.user_coords[name] = vec

        self.update_data_textboxes()
        self.refresh_calculator_dropdown()
        self.update_plot()
        self.tabview.set("PCA Map")

    # ------------------------------------------------------------------
    # 4. COORDINATE FILE TAB
    # ------------------------------------------------------------------
    def build_data_tab(self):
        frame = ctk.CTkFrame(self.tab_data)
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        top_bar = ctk.CTkFrame(frame)
        top_bar.pack(fill="x", padx=10, pady=10)

        # "Load .ppc file" Button (opens file picker dialog)
        btn_browse_ppc = ctk.CTkButton(
            top_bar,
            text="Load .ppc file",
            fg_color="#2b5b84",
            hover_color="#1e3f5c",
            command=self.browse_and_load_ppc,
        )
        btn_browse_ppc.pack(side="left", padx=10)

        btn_load = ctk.CTkButton(
            top_bar, text="Reload phenotypes.ppc", command=self.load_ppc_file
        )
        btn_load.pack(side="left", padx=10)

        btn_export_ppc = ctk.CTkButton(
            top_bar,
            text="Save Source to phenotypes.ppc",
            fg_color="#8a2be2",
            hover_color="#6a1b9a",
            command=self.export_ppc,
        )
        btn_export_ppc.pack(side="left", padx=10)

        columns_frame = ctk.CTkFrame(frame)
        columns_frame.pack(fill="both", expand=True, padx=10, pady=5)

        col_pheno = ctk.CTkFrame(columns_frame)
        col_pheno.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(
            col_pheno,
            text="Source (Reference Phenotypes)",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=5)
        self.txt_phenotypes = ctk.CTkTextbox(col_pheno, height=350, font=("Courier", 11))
        self.txt_phenotypes.pack(fill="both", expand=True, padx=5, pady=5)

        col_you = ctk.CTkFrame(columns_frame)
        col_you.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(
            col_you,
            text="Target (User / Custom Coordinates)",
            font=("Helvetica", 14, "bold"),
            text_color="#ff0055",
        ).pack(pady=5)
        self.txt_you = ctk.CTkTextbox(col_you, height=350, font=("Courier", 11))
        self.txt_you.pack(fill="both", expand=True, padx=5, pady=5)

        btn_apply = ctk.CTkButton(
            frame,
            text="Apply All Coordinates to PCA Map",
            command=self.apply_manual_coordinates,
        )
        btn_apply.pack(pady=10)

        self.update_data_textboxes()

    def browse_and_load_ppc(self):
        file_path = filedialog.askopenfilename(
            title="Select Encrypted Phenotype Database (.ppc)",
            filetypes=[("PhenoPlot Coordinates", "*.ppc"), ("All Files", "*.*")]
        )
        if file_path:
            loaded_data = pheno_core.load_phenotypes(file_path)
            if loaded_data:
                self.phenotype_coords = loaded_data
                self.update_data_textboxes()
                self.refresh_calculator_dropdown()
                self.update_plot()
                print(f"[+] Loaded custom database from: {file_path}")

    def export_ppc(self):
        self.apply_manual_coordinates()
        pheno_core.save_phenotypes(self.phenotype_coords, "phenotypes.ppc")

    def update_data_textboxes(self):
        self.txt_phenotypes.delete("1.0", "end")
        for name, vec in self.phenotype_coords.items():
            formatted_line = pheno_core.format_coordinate(name, vec)
            self.txt_phenotypes.insert("end", f"{formatted_line}\n")

        self.txt_you.delete("1.0", "end")
        for name, vec in self.user_coords.items():
            formatted_line = pheno_core.format_coordinate(name, vec)
            self.txt_you.insert("end", f"{formatted_line}\n")

    def parse_textbox_to_dict(self, textbox_widget):
        text = textbox_widget.get("1.0", "end").strip()
        lines = text.split("\n")
        result_dict = {}
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].replace('"', "").replace("'", "").strip()
                try:
                    coords = pheno_core.parse_coordinate_string(parts[1])
                    result_dict[name] = coords
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
        return result_dict

    def apply_manual_coordinates(self):
        self.phenotype_coords = self.parse_textbox_to_dict(self.txt_phenotypes)
        self.user_coords = self.parse_textbox_to_dict(self.txt_you)

        self.refresh_calculator_dropdown()
        self.update_plot()
        self.tabview.set("PCA Map")

    def load_ppc_file(self):
        if os.path.exists("phenotypes.ppc"):
            self.phenotype_coords = pheno_core.load_phenotypes("phenotypes.ppc")
            self.update_data_textboxes()
            self.refresh_calculator_dropdown()
            self.update_plot()

    # ------------------------------------------------------------------
    # 5. MIDPOINT SIMULATOR TAB
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
        out_name = self.entry_mid_name.get().strip() or "Midpoint"

        all_known = {**self.phenotype_coords, **self.user_coords}

        if input_a in all_known:
            vec_a = all_known[input_a]
        else:
            vec_a = pheno_core.parse_coordinate_string(input_a)

        if input_b in all_known:
            vec_b = all_known[input_b]
        else:
            vec_b = pheno_core.parse_coordinate_string(input_b)

        mid_vec = pheno_core.calculate_coordinate_midpoint(vec_a, vec_b)
        self.user_coords[out_name] = mid_vec

        self.update_data_textboxes()
        self.refresh_calculator_dropdown()
        self.update_plot()
        self.tabview.set("PCA Map")


if __name__ == "__main__":
    app = PhenoApp()
    app.mainloop()
