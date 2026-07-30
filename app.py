import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pheno_core
import tempfile
import os

# ------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PhenoPlot Web",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .signature {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-style: italic;
        color: #666666;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "phenotype_coords" not in st.session_state:
    st.session_state["phenotype_coords"] = pheno_core.load_phenotypes("phenotypes.ppc")

if "user_coords" not in st.session_state:
    st.session_state["user_coords"] = {}


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def plot_pca_map():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120, facecolor="#121212")
    ax.set_facecolor("#1b1b1b")

    valid_builtin = {
        k: v for k, v in st.session_state["phenotype_coords"].items() if len(v) == 13
    }
    valid_user = {
        k: v for k, v in st.session_state["user_coords"].items() if len(v) == 13
    }

    if not valid_builtin:
        st.warning("No valid reference phenotype data loaded.")
        return fig

    pheno_names = list(valid_builtin.keys())
    raw_matrix = [
        pheno_core.transform_features(valid_builtin[name]) for name in pheno_names
    ]
    data_matrix = np.array(raw_matrix)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    weighted_data = scaled_data * pheno_core.TRAIT_WEIGHTS

    pca = PCA(n_components=2)
    pheno_coords_2d = pca.fit_transform(weighted_data)

    # Reference Points
    ax.scatter(
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

        ax.annotate(
            name,
            (pheno_coords_2d[i, 0], pheno_coords_2d[i, 1]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=6,
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

    # User Points
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
        user_2d = np.sum(top_2d_coords * normalized_weights[:, np.newaxis], axis=0)

        ax.scatter(
            user_2d[0],
            user_2d[1],
            color="#ff0055",
            s=120,
            marker="*",
            edgecolors="#ffffff",
            linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            user_name,
            (user_2d[0], user_2d[1]),
            color="#ff0055",
            fontsize=8,
            fontweight="bold",
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
        )

    ax.grid(True, linestyle=":", alpha=0.15, color="#ffffff")
    ax.axhline(0, color="#444444", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="#444444", linestyle="-", linewidth=0.8, alpha=0.4)

    ax.set_title(
        "GLOBAL PHENOTYPE PCA SPACE MAP",
        fontsize=12,
        fontweight="bold",
        color="#00f0ff",
        loc="left",
    )
    ax.set_xlabel(
        "Principal Component 1 (Primary Morphological Variance)",
        color="#888888",
        fontsize=8,
    )
    ax.set_ylabel(
        "Principal Component 2 (Cranial Metrics & Pigmentation)",
        color="#888888",
        fontsize=8,
    )
    ax.tick_params(colors="#888888", labelsize=7)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Navigation / Tabs
# ------------------------------------------------------------------
st.title("PhenoPlot Web")

tab_plot, tab_calc, tab_data, tab_midpoint = st.tabs(
    ["PCA Map", "Phenotype Calculator", "Coordinate file", "Midpoint simulator"]
)

# ------------------------------------------------------------------
# TAB 1: PCA MAP
# ------------------------------------------------------------------
with tab_plot:
    st.subheader("PhenoPlot PCA Map")
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("Refresh Map", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.metric("Loaded References", len(st.session_state["phenotype_coords"]))
        st.metric("Custom Targets", len(st.session_state["user_coords"]))

    with col2:
        fig = plot_pca_map()
        st.pyplot(fig, clear_figure=True)

# ------------------------------------------------------------------
# TAB 2: PHENOTYPE CALCULATOR
# ------------------------------------------------------------------
with tab_calc:
    st.subheader("Phenotype Calculator Engine")

    all_known_names = list(st.session_state["user_coords"].keys()) + list(
        st.session_state["phenotype_coords"].keys()
    )

    c1, c2, c3 = st.columns([2, 3, 1])

    with c1:
        selected_profile = st.selectbox(
            "Select Target Profile:",
            options=all_known_names
            if all_known_names
            else ["No profiles available"],
        )

    with c2:
        raw_input_str = st.text_input(
            "OR Raw Coordinate:",
            placeholder='"Sample": [0.4512, 0.1241, ...]',
        )

    with c3:
        st.write(" ")
        st.write(" ")
        run_calc = st.button("Calculate Matches", use_container_width=True)

    if run_calc:
        all_coords = {
            **st.session_state["user_coords"],
            **st.session_state["phenotype_coords"],
        }
        target_vec = None
        display_name = ""

        if raw_input_str.strip():
            try:
                display_name, target_vec = pheno_core.parse_coordinate_with_name(
                    raw_input_str
                )
            except Exception as e:
                st.error(f"Error parsing input coordinates: {e}")
        elif selected_profile in all_coords:
            target_vec = all_coords[selected_profile]
            display_name = selected_profile

        if target_vec:
            pheno_names = list(st.session_state["phenotype_coords"].keys())
            raw_matrix = [
                pheno_core.transform_features(
                    st.session_state["phenotype_coords"][name]
                )
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

            st.markdown(f"### Results for: `{display_name}`")

            res_table = [
                {
                    "Rank": idx,
                    "Phenotype": p_name,
                    "Similarity Score": f"{sim:.1f}%",
                    "13D Distance": f"{dist:.2f}",
                }
                for idx, (p_name, sim, dist) in enumerate(results[:15], 1)
            ]
            st.dataframe(res_table, use_container_width=True)

# ------------------------------------------------------------------
# TAB 3: COORDINATE FILE WORKSPACE
# ------------------------------------------------------------------
with tab_data:
    st.subheader("Coordinate File Workspace")

    col_btn1, col_btn2, col_patreon = st.columns([2, 2, 3])

    with col_btn1:
        uploaded_files = st.file_uploader(
            "Load .ppc file(s):", type=["ppc"], accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".ppc"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loaded = pheno_core.load_phenotypes(tmp_path)
                if loaded:
                    st.session_state["phenotype_coords"].update(loaded)
                os.remove(tmp_path)
            st.success("Loaded custom .ppc file(s) successfully!")

    with col_btn2:
        st.write(" ")
        st.write(" ")
        if st.button("Reload phenotypes.ppc", use_container_width=True):
            st.session_state["phenotype_coords"] = pheno_core.load_phenotypes(
                "phenotypes.ppc"
            )
            st.session_state["user_coords"].clear()
            st.rerun()

    with col_patreon:
        st.write(" ")
        st.write(" ")
        st.link_button(
            "❤️ Get your custom coordinates here",
            "https://www.patreon.com/cw/PhenoPlot",
            use_container_width=True,
        )

    st.markdown("---")
    c_source, c_target = st.columns(2)

    with c_source:
        st.markdown("#### Source (Reference Phenotypes)")
        source_lines = [
            pheno_core.format_coordinate(k, v)
            for k, v in st.session_state["phenotype_coords"].items()
        ]
        st.text_area(
            "Source View (Read-Only)",
            value="\n".join(source_lines),
            height=350,
            disabled=True,
        )

    with c_target:
        st.markdown("#### Target (User / Custom Coordinates)")
        target_lines = [
            pheno_core.format_coordinate(k, v)
            for k, v in st.session_state["user_coords"].items()
        ]
        target_input = st.text_area(
            "Edit Target Coordinates",
            value="\n".join(target_lines),
            height=350,
        )

        if st.button(
            "Apply Target Coordinates to PCA Map & Calculator",
            use_container_width=True,
        ):
            new_targets = {}
            for line in target_input.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    parts = line.split(":", 1)
                    name = parts[0].replace('"', "").replace("'", "").strip()
                    try:
                        coords = pheno_core.parse_coordinate_string(parts[1])
                        new_targets[name] = coords
                    except Exception as e:
                        st.error(f"Error parsing line '{line}': {e}")
            st.session_state["user_coords"] = new_targets
            st.success("Target coordinates updated!")
            st.rerun()

# ------------------------------------------------------------------
# TAB 4: MIDPOINT SIMULATOR
# ------------------------------------------------------------------
with tab_midpoint:
    st.subheader("Midpoint Simulator")

    mid_col1, mid_col2 = st.columns(2)

    with mid_col1:
        input_a = st.text_input(
            "Set A (Name or Coordinate):",
            placeholder='e.g. Hallstatt OR "SampleA": [0.4512, 0.1241, ...]',
        )
    with mid_col2:
        input_b = st.text_input(
            "Set B (Name or Coordinate):",
            placeholder='e.g. Borreby OR "SampleB": [0.4512, 0.1241, ...]',
        )

    mid_name = st.text_input(
        "Simulated Coordinate Name:", value="Simulated_Midpoint"
    )

    if st.button("Calculate Midpoint & Plot", use_container_width=True):
        all_known = {
            **st.session_state["phenotype_coords"],
            **st.session_state["user_coords"],
        }
        vec_a, vec_b = None, None

        try:
            vec_a = (
                all_known[input_a.strip()]
                if input_a.strip() in all_known
                else pheno_core.parse_coordinate_string(input_a)
            )
            vec_b = (
                all_known[input_b.strip()]
                if input_b.strip() in all_known
                else pheno_core.parse_coordinate_string(input_b)
            )
        except Exception as e:
            st.error(f"Error parsing midpoint inputs: {e}")

        if vec_a and vec_b:
            mid_vec = pheno_core.calculate_coordinate_midpoint(vec_a, vec_b)
            st.session_state["user_coords"][mid_name] = mid_vec
            st.success(f"Midpoint '{mid_name}' added to map!")
            st.rerun()

# Footer Signature
st.markdown('<div class="signature">By Chuggy</div>', unsafe_allow_html=True)
