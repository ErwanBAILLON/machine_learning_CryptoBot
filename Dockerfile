# Utiliser une image de base Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires (optionnel)
# Par exemple, pour TA-Lib, il peut être nécessaire d'installer des bibliothèques système
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Copier le fichier requirements.txt dans le conteneur
COPY bot/requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application dans le conteneur
COPY bot/ .

# Définir les variables d'environnement (optionnel)
# ENV API_KEY=your_api_key
# ENV API_SECRET=your_api_secret

# Définir le point d'entrée du conteneur
CMD ["python", "main.py"]
