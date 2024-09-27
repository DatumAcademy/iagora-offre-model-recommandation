from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('modelRec.pkl', 'rb') as file:
    model, vecteurs_data = pickle.load(file)

etudiants_df = vecteurs_data["etudiants_df"]
offres_df = vecteurs_data["offres_df"]

@app.route('/recommander', methods=['GET'])
def recommander():
    student_id = request.args.get('student_id')

    if not student_id:
        return jsonify({"error": "student_id est requis"}), 400

    student_id = int(student_id)
    donnees_etudiant = etudiants_df[etudiants_df['numETU'] == student_id]
    if donnees_etudiant.empty:
        return jsonify({"error": "Étudiant non trouvé"}), 404

    vecteur_competences_etudiant = np.array(donnees_etudiant['vecteur_competences'].tolist())[0]
    experience_etudiant = donnees_etudiant['experience'].values[0]
    langue_etudiant = donnees_etudiant['langue'].values[0]

    matrice_competences_offres = np.array(offres_df['vecteur_competences'].tolist())

    X_pred_list = []
    for i, offre_vecteur_competences in enumerate(matrice_competences_offres):
        experience_min_offre = offres_df.iloc[i]['experience_min']
        langue_offre = offres_df.iloc[i]['langue']
        langue_match = 1 if langue_etudiant == langue_offre else 0
        X_pred = np.hstack([vecteur_competences_etudiant, offre_vecteur_competences,
                            [experience_etudiant, experience_min_offre, langue_match]])
        X_pred_list.append(X_pred)

    X_pred_array = np.array(X_pred_list)
    scores = model.predict_proba(X_pred_array)[:, 1]

    top_offers = [(offres_df.iloc[i], scores[i]) for i in range(len(scores))]

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
    app.run(host="0.0.0.0", port=5000, debug=True)
