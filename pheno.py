import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    "hypereuroprosopic": 76.0,
    "euryprosopic": 81.0,
    "mesoprosopic": 86.0,
    "leptoprosopic": 91.0,
    "hyperleptoprosopic": 96.0
}

SOMATOTYPE_DICT = {
    "ectomorph": 1.0,
    "mesomorph": 2.0,
    "endomorph": 3.0
}

PHENOTYPES = {
    "Nordic": [72.5, 72.5, 60.0, 91.0, 5, 1, 5, 180, 1.5],
    "Alpine": [82.5, 72.5, 77.0, 81.0, 3, 2, 3, 165, 3.0],
    "Dinaric": [87.5, 80.0, 60.0, 91.0, 2, 2, 2, 178, 2.0],
    "East Baltic": [82.5, 80.0, 77.0, 81.0, 6, 1, 6, 168, 3.0],
    "Mediterranean": [72.5, 72.5, 60.0, 91.0, 2, 3, 2, 165, 1.0]
}

TRAIT_WEIGHTS = np.array([1.5, 1.0, 1.5, 1.2, 1.0, 1.3, 1.0, 0.8, 1.0])

def get_input(prompt, options_dict):
    print(f"\nAvailable options: {', '.join(options_dict.keys())}")
    while True:
        val = input(prompt).strip().lower()
        if val in options_dict:
            return options_dict[val]
        print("Invalid choice. Please try again.")

def main():
    print("--- WELCOME TO THE PHENOTYPE CALCULATOR ---")
    
    user_cephalic = get_input("Select Cephalic Index: ", CEPHALIC_DICT)
    user_hl = get_input("Select Skull Height-Length: ", HEIGHT_LENGTH_DICT)
    user_nasal = get_input("Select Nasal Index: ", NASAL_DICT)
    user_facial = get_input("Select Facial Index: ", FACIAL_DICT)
    
    print("\n--- Pigmentation & Body ---")
    user_hair = float(input("Hair color (1=Ginger/Red, 2=Black, 3=Dark Brown, 4=Light Brown, 5=Blonde, 6=Platinum Blonde): "))
    user_skin = float(input("Skin color (Fitzpatrick scale 1-6): "))
    user_eye = float(input("Eye color (1=Dark Brown, 2=Brown, 3=Hazel, 4=Green, 5=Blue, 6=Light Blue/Grey): "))
    user_height = float(input("Height in cm (e.g., 180): "))
    user_somato = get_input("Select Somatotype: ", SOMATOTYPE_DICT)
    
    user_features = [user_cephalic, user_hl, user_nasal, user_facial, user_hair, user_skin, user_eye, user_height, user_somato]
    
    pheno_names = list(PHENOTYPES.keys())
    data_matrix = np.array([PHENOTYPES[name] for name in pheno_names] + [user_features])
    
    weighted_data = data_matrix * TRAIT_WEIGHTS
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(weighted_data)
    
    pca = PCA(n_components=2)
    pca_coordinates = pca.fit_transform(scaled_data)
    
    pheno_coords = pca_coordinates[:-1]
    user_coords = pca_coordinates[-1]
    
    distances = np.linalg.norm(pheno_coords - user_coords, axis=1)
    
    inv_distances = 1.0 / (distances + 1e-5)
    similarities = (inv_distances / np.sum(inv_distances)) * 100
    
    results = sorted(zip(pheno_names, similarities), key=lambda x: x[1], reverse=True)
    
    print("\n======================================")
    print("           YOUR PCA RESULTS           ")
    print("======================================")
    print(f"Your position in PCA space (X, Y): [{user_coords[0]:.2f}, {user_coords[1]:.2f}]\n")
    print("Your closest phenotype matches:")
    
    for name, pct in results:
        print(f"- {name}: {pct:.1f}% similarity")
    print("======================================")

if __name__ == "__main__":
    main()

