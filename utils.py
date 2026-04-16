import os
import requests

def download_file(url, dest):
    print(f"Scarico: {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(r.content)
    except requests.RequestException as e:
        print(f'Errore download {url}: {e}')

    


def ensure_project_files(base_url: str, file_paths: list):
    try:
        import google.colab  # noqa
        running_in_colab = True
        print("Google Colab")
    except ImportError:
        running_in_colab = False
        print("Locale / Jupyter")

    if(not running_in_colab):
        return
    
    for relative_path in file_paths:
        local_path = os.path.abspath(relative_path)

        #Crea le cartelle se non esistono
        folder = os.path.dirname(local_path)
        if folder != "" and not os.path.exists(folder):
            print(f"Creo cartella: {folder}")
            os.makedirs(folder, exist_ok=True)

        if os.path.exists(local_path):
            print(f"File già presente: {relative_path}")
        else:
            print(f"File mancante: {relative_path}")
            file_url = f"{base_url.rstrip('/')}/{relative_path}"
            download_file(file_url, local_path)
    
    print("Controllo completato.")
