from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import threading

app = Flask(__name__)

API_URL = "https://iagora-offre-serveur.onrender.com/OffreServeur"


def get_offers():
    offers_response = requests.get(f"{API_URL}/search")
    if offers_response.status_code == 200:
        return offers_response.json()['data']['offers']
    else:
        return []


def get_students():
    students_response = requests.get(f"{API_URL}/student/list/listStudent/getAll")
    if students_response.status_code == 200:
        return students_response.json()['students']
    else:
        return []


print(get_students)


def extract_skills_vector(skills_list, all_skills):
    vector = [1 if skill in skills_list else 0 for skill in all_skills]
    return vector


def experience_similarity(student_exp, offer_min_exp):
    if student_exp >= offer_min_exp:
        return 1
    else:
        return student_exp / offer_min_exp


def language_similarity(student_lang, offer_lang):
    return 1 if student_lang == offer_lang else 0


def contract_similarity(student_contract_pref, offer_contract):
    return 1 if student_contract_pref == offer_contract else 0


@app.route('/recommander', methods=['GET'])
def recommander():
    student_id = request.args.get('student_id')

    if not student_id:
        return jsonify({"error": "student_id est requis"}), 400

    student_id = int(student_id)
    offres = get_offers()
    etudiants = get_students()

    toutes_les_competences = set()
    for offre in offres:
        competences_offre = offre['skills'].split(", ")
        toutes_les_competences.update(competences_offre)

    for etudiant in etudiants:
        competences_etudiant = etudiant['skills']
        toutes_les_competences.update(competences_etudiant)

    toutes_les_competences = list(toutes_les_competences)

    donnees_etudiants = []
    for etudiant in etudiants:
        vecteur_competences_etudiant = extract_skills_vector(etudiant['skills'], toutes_les_competences)
        donnees_etudiants.append({
            "numETU": etudiant['numETU'],
            "nom": f"{etudiant['first_name']} {etudiant['last_name']}",
            "vecteur_competences": vecteur_competences_etudiant,
            "experience": etudiant['experience'][0]['yearsexperience'] if etudiant['experience'] else 0,
            "langue": etudiant['language'][0]['label'] if etudiant['language'] else "Non spécifié"
        })

    donnees_offres = []
    for offre in offres:
        vecteur_competences_offre = extract_skills_vector(offre['skills'].split(", "), toutes_les_competences)
        donnees_offres.append({
            "offer_id": offre['id'],
            "label": offre['label'],
            "entreprise": offre['company'],
            "vecteur_competences": vecteur_competences_offre,
            "experience_min": offre['minexperience'],
            "langue": offre['language']['label'],
            "contrat": offre['contract']
        })

    etudiants_df = pd.DataFrame(donnees_etudiants)
    offres_df = pd.DataFrame(donnees_offres)

    print(etudiants_df.columns)

    donnees_etudiant = etudiants_df[etudiants_df['numETU'] == student_id]
    if donnees_etudiant.empty:
        return jsonify({"error": "Étudiant non trouvé"}), 404

    vecteur_competences_etudiant = np.array(donnees_etudiant['vecteur_competences'].tolist())
    experience_etudiant = donnees_etudiant['experience'].values[0]
    langue_etudiant = donnees_etudiant['langue'].values[0]

    matrice_competences_offres = np.array(offres_df['vecteur_competences'].tolist())
    similarite_competences = cosine_similarity(vecteur_competences_etudiant, matrice_competences_offres)

    similarite_experience = offres_df['experience_min'].apply(
        lambda x: experience_similarity(experience_etudiant, x)).values
    similarite_langue = offres_df['langue'].apply(lambda x: language_similarity(langue_etudiant, x)).values
    similarite_contrat = offres_df['contrat'].apply(lambda x: contract_similarity("Stagiaire", x)).values

    score_total = 0.4 * similarite_competences[
        0] + 0.3 * similarite_experience + 0.2 * similarite_langue + 0.1 * similarite_contrat

    meilleures_offres = offres_df.loc[np.argsort(-score_total)[:5], ['label', 'entreprise']]

    return jsonify({
        "student_id": student_id,
        "recommendations": meilleures_offres.to_dict(orient="records")
    })


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
