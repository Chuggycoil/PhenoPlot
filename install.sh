#!/bin/bash
echo "Installerar PhenoPCA..."

# Skapa virtuell miljö om den inte finns
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Aktivera och installera bibliotek
source .venv/bin/activate
pip install -r requirements.txt

# Hämta aktuell mappsökväg
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Skapa .desktop-filen för Linux
DESKTOP_FILE="$HOME/.local/share/applications/phenopca.desktop"

cat <<EOT > "$DESKTOP_FILE"
[Desktop Entry]
Type=Application
Name=PhenoPCA
Comment=Phenotype Calculator & PCA App
Exec=$APP_DIR/.venv/bin/python3 $APP_DIR/gui_main.py
Icon=$APP_DIR/app_logo.png
Terminal=false
Categories=Science;Education;
EOT

chmod +x "$DESKTOP_FILE"
echo "Installation klar! Du hittar nu PhenoPCA bland dina appar."
