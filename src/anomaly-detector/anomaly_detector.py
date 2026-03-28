import numpy as np

import pandas as pd

from sklearn.ensemble import IsolationForest

import joblib

import logging

import os

from datetime import datetime



logger = logging.getLogger(__name__)



class AnomalyDetector:

    """

    Détecteur d'anomalies basé sur Isolation Forest

    """

    

    def __init__(self, contamination=0.1, random_state=42):

        self.model = IsolationForest(

            contamination=contamination,

            random_state=random_state,

            n_estimators=100,

            max_samples='auto',

            verbose=0

        )

        self.feature_columns = ["mean", "std", "max", "min", "last_value", "trend"]

        self.service_info = None

        self.is_trained = False

        

    def train(self, X_scaled, features_df):

        """

        Entraîne le modèle

        """

        if X_scaled is None or len(X_scaled) < 10:

            logger.error("Pas assez de données pour l'entraînement")

            return False

        

        logger.info(f"Entraînement avec {len(X_scaled)} échantillons...")

        

        try:

            self.model.fit(X_scaled)

            self.service_info = features_df[["service", "metric_type"]].copy()

            self.is_trained = True

            

            # Évaluer

            predictions = self.model.predict(X_scaled)

            n_anomalies = np.sum(predictions == -1)

            logger.info(f"✅ Entraînement terminé - Anomalies détectées: {n_anomalies}/{len(predictions)}")

            return True

            

        except Exception as e:

            logger.error(f"Erreur entraînement: {e}")

            return False

    

    def predict(self, X_scaled, features_df):

        """

        Prédit les anomalies

        """

        if not self.is_trained or X_scaled is None:

            return None, None

        

        predictions = self.model.predict(X_scaled)

        scores = self.model.score_samples(X_scaled)

        

        results = features_df.copy()

        results["is_anomaly"] = predictions == -1

        results["anomaly_score"] = scores

        

        # Normaliser le score (0 à 1)

        min_score = scores.min()

        max_score = scores.max()

        if max_score > min_score:

            results["anomaly_score_normalized"] = (scores - min_score) / (max_score - min_score)

        else:

            results["anomaly_score_normalized"] = 0

        

        anomalies = results[results["is_anomaly"]]

        

        return results, anomalies

    

    def analyze_current_data(self, X_scaled, features_df):

        """

        Analyse et retourne les anomalies

        """

        if not self.is_trained:

            logger.warning("Modèle non entraîné")

            return []

        

        results, anomalies = self.predict(X_scaled, features_df)

        

        if anomalies is None or anomalies.empty:

            return []

        

        anomaly_list = []

        for _, row in anomalies.iterrows():

            anomaly = {

                "timestamp": datetime.now().isoformat(),

                "service": row["service"],

                "metric_type": row["metric_type"],

                "severity": self._calculate_severity(row["anomaly_score"]),

                "current_values": {

                    "mean": float(row["mean"]),

                    "std": float(row["std"]),

                    "max": float(row["max"]),

                    "min": float(row["min"]),

                    "last_value": float(row["last_value"]),

                    "trend": float(row["trend"])

                },

                "anomaly_score": float(row["anomaly_score"]),

                "confidence": float(1 - row["anomaly_score_normalized"]),

                "recommendation": self._generate_recommendation(row)

            }

            anomaly_list.append(anomaly)

        

        return anomaly_list

    

    def _calculate_severity(self, score):

        if score < -0.5:

            return "CRITICAL"

        elif score < -0.3:

            return "HIGH"

        elif score < -0.1:

            return "MEDIUM"

        else:

            return "LOW"

    

    def _generate_recommendation(self, row):

        service = row["service"]

        metric = row["metric_type"]

        trend = row["trend"]

        

        recommendations = {

            "cpu": f"⚠️ CPU élevé sur {service} (tendance: +{trend:.1%}) → Vérifier le scaling",

            "memory": f"⚠️ Mémoire élevée sur {service} (tendance: +{trend:.1%}) → Fuite mémoire possible",

        }

        return recommendations.get(metric, f"⚠️ Comportement anormal sur {service} ({metric})")

    

    def save_model(self, path="models/anomaly_detector.joblib"):

        try:

            os.makedirs(os.path.dirname(path), exist_ok=True)

            model_data = {"model": self.model, "feature_columns": self.feature_columns}

            joblib.dump(model_data, path)

            logger.info(f"✅ Modèle sauvegardé: {path}")

            return True

        except Exception as e:

            logger.error(f"Erreur sauvegarde: {e}")

            return False

    

    def load_model(self, path="models/anomaly_detector.joblib"):

        try:

            if not os.path.exists(path):

                return False

            model_data = joblib.load(path)

            self.model = model_data["model"]

            self.feature_columns = model_data["feature_columns"]

            self.is_trained = True

            logger.info(f"✅ Modèle chargé: {path}")

            return True

        except Exception as e:

            logger.error(f"Erreur chargement: {e}")

            return False
