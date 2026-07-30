import json
import os

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

HEIGHT_DICT = {
    "very short": 162.5,
    "short": 167.5,
    "rather short": 172.5,
    "medium": 177.5,
    "rather tall": 182.5,
    "tall": 187.5,
    "very tall": 192.5
}


def load_phenotypes():
    if not os.path.exists("phenotypes.json"):
        return {}
    try:
        with open("phenotypes.json", "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_phenotypes(data):
    with open("phenotypes.json", "w") as f:
        json.dump(data, f, indent=2)


def get_input(prompt, options_dict):
    print(f"\nAvailable options: {', '.join(options_dict.keys())}")
    while True:
        val = input(prompt).strip().lower()
        if val in options_dict:
            return options_dict[val]
        print("Invalid choice. Please try again.")


def main():
    print("--- PHENOTYPE DATABASE MANAGER ---")
    print("Type 'exit' as the phenotype name at any time to save and quit.")
    
    while True:
        phenotypes = load_phenotypes()
        
        print("\n======================================")
        name = input("Enter the name of the new phenotype: ").strip()
        
        if name.lower() == 'exit':
            print("\nExiting database manager. All changes saved!")
            break
            
        if not name:
            print("Name cannot be empty.")
            continue
            
        if name in phenotypes:
            overwrite = input(f"'{name}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Skipped.")
                continue

        # LEVEL 1: MACRO-MORPHOLOGY
        print("\n--- LEVEL 1: MACRO-MORPHOLOGY ---")
        progn = get_input("Select Facial Prognathism: ", PROGNATHISM_DICT)
        htext = get_input("Select Hair Texture: ", HAIR_TEXTURE_DICT)
        eyeshape = get_input("Select Eye Shape / Fold: ", EYE_SHAPE_DICT)

        # LEVEL 2: CRANIAL & FACIAL STRUCTURE
        print("\n--- LEVEL 2: CRANIAL & FACIAL STRUCTURE ---")
        fwhr = get_input("Select Facial Width-to-Height Ratio (FWHR): ", FWHR_DICT)
        cephalic = get_input("Select Cephalic Index: ", CEPHALIC_DICT)
        hl = get_input("Select Skull Height-Length: ", HEIGHT_LENGTH_DICT)
        nasal = get_input("Select Nasal Index: ", NASAL_DICT)
        facial = get_input("Select Facial Index: ", FACIAL_DICT)
        
        # LEVEL 3: PIGMENTATION
        print("\n--- LEVEL 3: PIGMENTATION ---")
        while True:
            try:
                hair = float(input("Hair color (1=Black, 2=Dark Brown, 3=Light Brown, 4=Ginger/Red, 5=Blonde, 6=Platinum Blonde): "))
                if 1 <= hair <= 6: 
                    break
                print("Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print("\nSkin color (Fitzpatrick scale 1-6):")
        print(" 1 = Very Pale\n 2 = Fair / Light\n 3 = Medium / Olive\n 4 = Dark Brown / Olive\n 5 = Dark Brown / Black\n 6 = Deeply Pigmented / Black")
        while True:
            try:
                skin = float(input("Enter choice (1-6): "))
                if 1 <= skin <= 6: 
                    break
                print("Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        print("")
        while True:
            try:
                eye = float(input("Eye color (1=Dark Brown, 2=Brown, 3=Hazel, 4=Green, 5=Blue, 6=Light Blue/Grey): "))
                if 1 <= eye <= 6: 
                    break
                print("Please enter a number between 1 and 6.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # LEVEL 4: STATURE & BODY BUILD
        print("\n--- LEVEL 4: STATURE & BODY BUILD ---")
        height = get_input("Select Height level: ", HEIGHT_DICT)
        somato = get_input("Select Somatotype: ", SOMATOTYPE_DICT)
        
        # Array Structure (13 elements total)
        phenotypes[name] = [
            progn, htext, eyeshape, fwhr, cephalic, hl, nasal, 
            facial, hair, skin, eye, height, somato
        ]
        
        save_phenotypes(phenotypes)
        print(f"\n[Success] '{name}' has been added to phenotypes.json with 13 features!")


if __name__ == "__main__":
    main()
