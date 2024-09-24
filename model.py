import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import re
import os

API_URL = "https://iagora-offre-serveur.onrender.com/OffreServeur"

def get_offers():
    offers_response = requests.get(f"{API_URL}/search?pageSize=3000")
    if offers_response.status_code == 200:
        return offers_response.json().get('data', {}).get('offers', [])
    else:
        return []

def get_students():
    students_response = requests.get(f"{API_URL}/student/list/listStudent/getAll?pageSize=500")
    if students_response.status_code == 200:
        return students_response.json().get('students', [])
    else:
        return []

def normalize_skill(skill):
    skill = skill.lower().strip()
    skill = re.sub(r'[^a-z0-9]', '', skill)
    corrections = {
        'reactjs': 'react',
        'vuejs': 'vue',
        'nodejs': 'nodejs',
        'expressjs': 'express',
        'c++': 'cpp',
        'c#': 'csharp',
        'aspnet': 'aspnet',
        'springboot': 'spring',
        'typescript': 'ts',
        'javascript': 'js',
        'html5': 'html',
        'css3': 'css',
        'postgresql': 'postgres',
        'sqlserver': 'sql',
        'python3': 'python',
        'mongodb': 'mongodb',
        'flask': 'flask',
        'django': 'django',
        'ruby': 'ruby',
        'ruby on rails': 'rails'
    }
    return corrections.get(skill, skill)

def extract_skills_vector(skills_list, all_skills):
    normalized_skills_list = [normalize_skill(skill) for skill in skills_list]
    vector = [1 if normalize_skill(skill) in normalized_skills_list else 0 for skill in all_skills]
    return vector

def prepare_data():
    offres = get_offers()
    etudiants = get_students()

    toutes_les_competences = set()
    for offre in offres:
        competences_offre = offre.get('skills', '').split(", ")
        toutes_les_competences.update([normalize_skill(skill) for skill in competences_offre])

    for etudiant in etudiants:
        competences_etudiant = etudiant.get('skills', [])
        toutes_les_competences.update([normalize_skill(skill) for skill in competences_etudiant])

    toutes_les_competences = list(toutes_les_competences)

    donnees_etudiants = []
    for etudiant in etudiants:
        experience = etudiant.get('experience', [])
        yearsexperience = experience[0].get('yearsexperience', 0) if experience else 0

        vecteur_competences_etudiant = extract_skills_vector(etudiant.get('skills', []), toutes_les_competences)
        donnees_etudiants.append({
            "numETU": etudiant['numETU'],
            "vecteur_competences": vecteur_competences_etudiant,
            "experience": yearsexperience,
            "langue": etudiant['language'][0]['label'] if etudiant.get('language') and len(
                etudiant['language']) > 0 else 'Non spécifié'
        })

    donnees_offres = []
    for offre in offres:
        vecteur_competences_offre = extract_skills_vector(offre.get('skills', '').split(", "), toutes_les_competences)
        donnees_offres.append({
            "offer_id": offre['id'],
            "label": offre['label'],
            "entreprise": offre['company'],
            "vecteur_competences": vecteur_competences_offre,
            "experience_min": offre.get('minexperience', 0),
            "langue": offre.get('language', {}).get('label', 'Non spécifié'),
            "contrat": offre.get('contract', 'Non spécifié')
        })

    return pd.DataFrame(donnees_etudiants), pd.DataFrame(donnees_offres), toutes_les_competences

def train_model():
    etudiants_df, offres_df, toutes_les_competences = prepare_data()

    data = []
    for _, etudiant in etudiants_df.iterrows():
        for _, offre in offres_df.iterrows():
            data.append({
                "student_id": etudiant["numETU"],
                "offer_id": offre["offer_id"],
                "vecteur_competences_etudiant": etudiant["vecteur_competences"],
                "vecteur_competences_offre": offre["vecteur_competences"],
                "experience_etudiant": etudiant["experience"],
                "experience_min_offre": offre["experience_min"],
                "langue_etudiant": etudiant["langue"],
                "langue_offre": offre["langue"],
                "applied": np.random.randint(0, 2)
            })

    df = pd.DataFrame(data)

    X = np.hstack([
        np.vstack(df['vecteur_competences_etudiant']),
        np.vstack(df['vecteur_competences_offre']),
        df[['experience_etudiant', 'experience_min_offre']].values,
        (df['langue_etudiant'] == df['langue_offre']).values.reshape(-1, 1)
    ])

    y = df['applied']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        return None

if __name__ == '__main__':
    if os.path.exists('model.pkl'):
        model = load_model()
    else:
        model = train_model()
