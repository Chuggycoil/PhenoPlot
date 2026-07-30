import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- DICTIONARIES FOR GLOBAL TYPOLOGY ---

PROGNATHISM_DICT = {
    "orthognathic": 1.0,   
    "mesognathic": 2.2,    
    "prognathic": 4.5      
}

HAIR_TEXTURE_DICT = {
    "coarse_straight": 1.0,
    "straight": 4.0,
    "wavy": 5.0,
    "curly": 6.0,
    "kinky": 9.5,
    "peppercorn": 12.0
}

EYE_SHAPE_DICT = {
    "standard_open": 1.0,        
    "slight_fold": 2.2,          
    "full_epicanthic_fold": 4.5   
}

FWHR_DICT = {
    "gracile_narrow": 1.6,     
    "medium": 1.8,
    "cromagnid_broad": 2.1     
}

CEPHALIC_DICT = {
    "hyperdolichocephalic": 67.5,
    "dolichocephalic": 72.5,
    "mesocephalic": 77.5,
    "brachycephalic": 82.5,
    "hyperbrachycephalic": 87.5
}

HEIGHT_LENGTH_DICT = {
    "chamaecranic": 65.0,
    "orthocranic": 72.5,
    "hypsicranic": 80.0
}

NASAL_DICT = {
    "hyperleptorrhine": 50.0,
    "leptorrhine": 60.0,
    "mesorrhine": 77.0,
    "platyrrhine": 92.0,
    "hyperplatyrrhine": 105.0
}

FACIAL_DICT = {
    "euryprosopic": 81.0,
    "mesoprosopic": 86.0,
    "leptoprosopic": 91.0
}

SOMATOTYPE_DICT = {
    "ectomorph": 1.0,
    "mesomorph": 2.0,
    "endomorph": 3.0
}

# --- REBALANCED TRAIT WEIGHTS ---
TRAIT_WEIGHTS = np.array([
    3.2,  # Prognathism
    3.5,  # Hair Texture
    4.2,  # Eye Shape
    2.4,  # FWHR
    3.0,  # Cephalic Index
    2.5,  # Height-Length
    3.5,  # Nasal Index
    2.2,  # Facial Index
    2.0,  # Hair Color
    3.8,  # Skin Color
    2.0,  # Eye Color
    0.5,  # Height
    0.5   # Somatotype
])


def transform_features(raw_features):
    transformed = list(raw_features)
    
    nasal_val = transformed[6]
    if nasal_val > 72.0:
        transformed[6] = 72.0 + ((nasal_val - 72.0) * 1.5)

    skin_val = transformed[9]
    if skin_val >= 3.0:
        transformed[9] = 2.0 + ((skin_val - 2.0) * 3.8)

    eye_shape_val = transformed[2]
    if eye_shape_val >= 1.8:
        transformed[2] = 1.0 + ((eye_shape_val - 1.0) * 2.2)

    return transformed


def load_phenotypes():
    try:
        with open("phenotypes.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: phenotypes.json not found! Please check your folder.")
        exit(1)


def get_input(prompt, options_dict):
    print(f"\nAvailable options: {', '.join(options_dict.keys())}")
    while True:
        val = input(prompt).strip().lower()
        if val in options_dict:
            return options_dict[val]
        print("Invalid choice. Please try again.")


def generate_png_plot(pheno_names, pheno_coords, user_coords=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 13), facecolor='#121212')
    ax.set_facecolor('#1b1b1b')
    
    ax.grid(True, linestyle=':', alpha=0.15, color='#ffffff')
    ax.axhline(0, color='#444444', linestyle='-', linewidth=1, alpha=0.4)
    ax.axvline(0, color='#444444', linestyle='-', linewidth=1, alpha=0.4)
    
    ax.scatter(pheno_coords[:, 0], pheno_coords[:, 1], 
               color='#00f0ff', s=120, edgecolors='#ffffff', 
               linewidths=0.8, alpha=0.8, label='Reference Phenotypes', zorder=3)
    
    for i, name in enumerate(pheno_names):
        angle = (i * 53) % 360
        rad = np.radians(angle)
        dx = 15 * np.cos(rad)
        dy = 15 * np.sin(rad)
        
        ax.annotate(name, (pheno_coords[i, 0], pheno_coords[i, 1]), 
                    textcoords="offset points", xytext=(dx, dy), ha='center', va='center',
                    color='#e0e0e0', fontsize=8, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#121212", ec="#333333", alpha=0.7))
        
    if user_coords is not None:
        ax.scatter(user_coords[0], user_coords[1], 
                   color='#ff0055', s=380, marker='*', edgecolors='#ffffff', 
                   linewidths=1.5, label='YOU', zorder=5)
        
        ax.annotate('YOU', (user_coords[0], user_coords[1]), 
                    textcoords="offset points", xytext=(0, -22), ha='center', 
                    color='#ff0055', fontsize=11, fontweight='bold')
    
    ax.set_title('GLOBAL PHENOTYPE PCA SPACE MAP', 
                 fontsize=15, fontweight='bold', pad=20, color='#00f0ff', loc='left')
    ax.set_xlabel('Principal Component 1 (Primary Morphological Variance)', fontsize=10, color='#888888', labelpad=10)
    ax.set_ylabel('Principal Component 2 (Cranial Metrics & Pigmentation)', fontsize=10, color='#888888', labelpad=10)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
        
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#121212', edgecolor='#444444', fontsize=9)
    plt.setp(legend.get_texts(), color='#ffffff')
    
    output_filename = "pca_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[Visual] PCA plot saved as '{output_filename}'.")


def run_questions():
    print("\n--- MACRO-MORPHOLOGY ---")
    user_progn = get_input("Select Facial Prognathism: ", PROGNATHISM_DICT)
    user_htext = get_input("Select Hair Texture: ", HAIR_TEXTURE_DICT)
    user_eyeshape = get_input("Select Eye Shape / Fold: ", EYE_SHAPE_DICT)

    print("\n--- CRANIAL & FACIAL STRUCTURE ---")
    user_fwhr = get_input("Select FWHR (Facial Width-to-Height): ", FWHR_DICT)
    user_cephalic = get_input("Select Cephalic Index: ", CEPHALIC_DICT)
    user_hl = get_input("Select Skull Height-Length: ", HEIGHT_LENGTH_DICT)
    user_nasal = get_input("Select Nasal Index: ", NASAL_DICT)
    user_facial = get_input("Select Facial Index: ", FACIAL_DICT)

    print("\n--- PIGMENTATION ---")
    print("\nHair color options:")
    print(" 1 = Black\n 2 = Dark Brown\n 3 = Light Brown\n 4 = Red / Ginger\n 5 = Blonde\n 6 = Platinum Blonde")
    while True:
        try:
            user_hair = float(input("Select Hair color (1-6): "))
            if 1 <= user_hair <= 6: break
            print("Please enter a number between 1 and 6.")
        except ValueError: 
            print("Invalid input.")

    print("\nSkin color (Fitzpatrick scale 1-6):")
    print(" 1 = Very Pale\n 2 = Fair / Light\n 3 = Medium / Olive\n 4 = Dark Brown / Olive\n 5 = Dark Brown / Black\n 6 = Deeply Pigmented / Black")
    while True:
        try:
            user_skin = float(input("Select Skin color (1-6): "))
            if 1 <= user_skin <= 6: break
            print("Please enter a number between 1 and 6.")
        except ValueError: 
            print("Invalid input.")
            
    print("\nEye color options:")
    print(" 1 = Dark Brown\n 2 = Brown\n 3 = Hazel\n 4 = Green\n 5 = Blue\n 6 = Light Blue / Grey")
    while True:
        try:
            user_eye = float(input("Select Eye color (1-6): "))
            if 1 <= user_eye <= 6: break
            print("Please enter a number between 1 and 6.")
        except ValueError: 
            print("Invalid input.")

    print("\n--- STATURE & BODY BUILD ---")
    while True:
        try:
            user_height = float(input("Enter height in cm (e.g., 180): "))
            if 100 <= user_height <= 250: break
            print("Please enter height between 100 and 250 cm.")
        except ValueError: 
            print("Invalid input.")

    user_somato = get_input("Select Somatotype: ", SOMATOTYPE_DICT)
    
    return [
        user_progn, user_htext, user_eyeshape, user_fwhr,
        user_cephalic, user_hl, user_nasal, user_facial,
        user_hair, user_skin, user_eye, 
        user_height, user_somato
    ]


def main():
    phenotypes = load_phenotypes()
    
    print("==============================================")
    print("      GLOBAL PHENOTYPE PCA CALCULATOR         ")
    print("==============================================")
    print("Select Mode:")
    print(" 1. Full Questionnaire Mode (Answer all questions)")
    print(" 2. Quick Direct Plot (Plot database space instantly)")
    print(" 3. Select Preset Target (Test a specific phenotype as 'YOU')")
    
    mode = input("\nEnter choice (1, 2 or 3): ").strip()
    pheno_names = list(phenotypes.keys())
    
    raw_matrix = [transform_features(phenotypes[name]) for name in pheno_names]
    data_matrix = np.array(raw_matrix)
    
    # 1. Träna Scaler och PCA på referensdatabasen
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    weighted_data = scaled_data * TRAIT_WEIGHTS
    
    pca = PCA(n_components=2)
    pheno_coords = pca.fit_transform(weighted_data)
    
    if mode == '2':
        generate_png_plot(pheno_names, pheno_coords)
        return

    elif mode == '3':
        print("\nAvailable Preset Phenotypes:")
        for idx, name in enumerate(pheno_names, 1):
            print(f" {idx}. {name}")
        
        while True:
            try:
                choice = int(input("\nSelect number: "))
                if 1 <= choice <= len(pheno_names):
                    selected_name = pheno_names[choice - 1]
                    raw_user_features = phenotypes[selected_name]
                    print(f"\n[Test Mode] Running PCA with target set to '{selected_name}'...")
                    break
            except ValueError:
                pass
    else:
        raw_user_features = run_questions()
    
    # 2. Transformera användaren via samma vikter
    user_features = transform_features(raw_user_features)
    scaled_user = scaler.transform([user_features])[0]
    weighted_user = scaled_user * TRAIT_WEIGHTS
    
    # 3. Beräkna 13D-avstånd mot alla typer
    distances_13d = np.linalg.norm(weighted_data - weighted_user, axis=1)
    
    max_dist = np.max(distances_13d) if np.max(distances_13d) > 0 else 1.0
    similarities = np.exp(-distances_13d / (max_dist * 0.35)) * 100
    
    results = sorted(zip(pheno_names, similarities, pheno_coords), key=lambda x: x[1], reverse=True)
    top_results = results[:10]
    
    # 4. VIKTAT GENOMSNITT AV TOPP 5 (med #1 som dominant ankare)
    top_5_matches = top_results[:5]
    
    raw_pcts = np.array([pct for _, pct, _ in top_5_matches])
    
    # Exponentiell viktning med potens 4:
    # Gör att #1 väger extremt tungt, medan #2, #3, #4 och #5 bidrar med en mindre dragkraft
    weighted_pcts = raw_pcts ** 4  
    
    normalized_weights = weighted_pcts / np.sum(weighted_pcts)
    top_2d_coords = np.array([coords for _, _, coords in top_5_matches])
    
    user_coords = np.sum(top_2d_coords * normalized_weights[:, np.newaxis], axis=0)
    
    print("\n==============================================")
    print("             YOUR PCA RESULTS                 ")
    print("==============================================")
    print(f"Position in PCA space (X, Y): [{user_coords[0]:.2f}, {user_coords[1]:.2f}]\n")
    print("Top 10 closest phenotype matches:")
    for name, pct, _ in top_results:
        print(f"- {name}: {pct:.1f}% similarity")
    print("==============================================")

    generate_png_plot(pheno_names, pheno_coords, user_coords)


if __name__ == "__main__":
    main()
