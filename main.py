from flask import Flask, request, jsonify
import numpy as np
from model import load_model, prepare_data_cached

app = Flask(__name__)

model = load_model()
if model is None:
    print("Erreur : Le modèle n'a pas été chargé correctement.")
else:
    print("Le modèle a été chargé avec succès.")

@app.route('/recommander', methods=['GET'])
def recommander():
    student_id = request.args.get('student_id')

    if not student_id:
        return jsonify({"error": "student_id est requis"}), 400

    student_id = int(student_id)
    etudiants_df, offres_df, toutes_les_competences = prepare_data_cached()

    donnees_etudiant = etudiants_df[etudiants_df['numETU'] == student_id]
    if donnees_etudiant.empty:
        return jsonify({"error": "Étudiant non trouvé"}), 404

    vecteur_competences_etudiant = np.array(donnees_etudiant['vecteur_competences'].tolist())[0]
    experience_etudiant = donnees_etudiant['experience'].values[0]
    langue_etudiant = donnees_etudiant['langue'].values[0]

    matrice_competences_offres = np.array(offres_df['vecteur_competences'].tolist())

    top_offers = []

    for i, offre_vecteur_competences in enumerate(matrice_competences_offres):
        experience_min_offre = offres_df.iloc[i]['experience_min']
        langue_offre = offres_df.iloc[i]['langue']

        langue_match = 1 if langue_etudiant == langue_offre else 0

        X_pred = np.hstack([vecteur_competences_etudiant, offre_vecteur_competences, [experience_etudiant, experience_min_offre, langue_match]])

        score = model.predict_proba([X_pred])[0][1]
        top_offers.append((offres_df.iloc[i], score))

    top_offers = sorted(top_offers, key=lambda x: x[1], reverse=True)

    recommandations = [{
        "offer_id": offre['offer_id'],
        "label": offre['label'],
        "entreprise": offre['entreprise']
    } for offre, score in top_offers[:10]]

    return jsonify({
        "student_id": student_id,
        "recommendations": recommandations
    })

if __name__ == '__main__':
    print("Lancement de l'application Flask...")
    app.run(host="0.0.0.0", port=5000, debug=True)
