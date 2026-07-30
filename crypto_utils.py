# crypto_utils.py
import json
from cryptography.fernet import Fernet

SECRET_KEY = b'vE30U3G9g7wQ0F4S_1Y4E_K9X822aJ3m2-R1l1c_13M='

def save_ppc_file(data_dict: dict, ppc_path="phenotypes.ppc", key: bytes = SECRET_KEY):
    """Encrypts a Python dictionary directly into an encrypted .ppc file."""
    fernet = Fernet(key)
    json_bytes = json.dumps(data_dict, indent=2).encode('utf-8')
    encrypted_bytes = fernet.encrypt(json_bytes)
    with open(ppc_path, 'wb') as f:
        f.write(encrypted_bytes)
    print(f"[+] Successfully saved encrypted database -> {ppc_path}")

def load_ppc_file(ppc_path="phenotypes.ppc", key: bytes = SECRET_KEY) -> dict:
    """Decrypts a .ppc file directly into a Python dictionary in memory."""
    fernet = Fernet(key)
    with open(ppc_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted_bytes = fernet.decrypt(encrypted_data)
    return json.loads(decrypted_bytes.decode('utf-8'))
