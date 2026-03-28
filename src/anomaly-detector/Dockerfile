FROM python:3.9-slim



WORKDIR /app



# Installer les dépendances système

RUN apt-get update && apt-get install -y \

    gcc \

    && rm -rf /var/lib/apt/lists/*



# Copier les requirements

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt



# Copier le code

COPY *.py .

COPY models/ models/



# Créer le répertoire pour les modèles

RUN mkdir -p models



# Exposer le port

EXPOSE 5000



# Variable d'environnement pour le token

ENV PROMETHEUS_TOKEN=""



# Lancer l'application

CMD ["python", "api.py"]
