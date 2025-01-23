#!/bin/bash

# On vérifie si le fichier requirements.txt existe
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found!"
    exit 1
fi

# On vérifie si un environnement virtuel existe
if [ -d .venv ] || [ -d venv ] || [ -d env ]; then
    echo "Virtual environment already exists."
    if [ -d .venv ]; then
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
    elif [ -d venv ]; then
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
    elif [ -d env ]; then
        source env/bin/activate 2>/dev/null || source env/Scripts/activate
    fi
    exit 0
fi

# On vérifie l'OS pour créer et activer l'environnement virtuel
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    python3 -m venv .venv
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    python -m venv .venv
    source .venv/Scripts/activate
else
    echo "Unsupported OS"
    exit 1
fi

# On installe les dépendances
pip install -r requirements.txt

echo "Environment setup complete."
