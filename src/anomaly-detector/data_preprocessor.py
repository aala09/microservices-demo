import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

import logging

from datetime import datetime



logger = logging.getLogger(__name__)



class DataPreprocessor:
  

    """

    Prépare les données pour le modèle de détection d'anomalies

    """

    
    def save(self, path="models/preprocessor.joblib"):
        """Sauvegarde le préprocesseur"""
        try:
            import joblib
            preprocessor_data = {
                "scaler": self.scaler,
                "imputer": self.imputer,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(preprocessor_data, path)
            logger.info(f"✅ Préprocesseur sauvegardé: {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde préprocesseur: {e}")
            return False
    
    def load(self, path="models/preprocessor.joblib"):
        """Charge le préprocesseur"""
        try:
            import joblib
            if not os.path.exists(path):
                return False
            preprocessor_data = joblib.load(path)
            self.scaler = preprocessor_data["scaler"]
            self.imputer = preprocessor_data["imputer"]
            self.feature_columns = preprocessor_data["feature_columns"]
            self.is_fitted = preprocessor_data["is_fitted"]
            logger.info(f"✅ Préprocesseur chargé: {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement préprocesseur: {e}")
            return False
    def __init__(self):

        self.scaler = StandardScaler()

        self.imputer = SimpleImputer(strategy="mean")

        self.feature_columns = ["mean", "std", "max", "min", "last_value", "trend"]

        self.is_fitted = False

        

    def create_feature_vector(self, metrics_data):

        """

        Transforme les métriques en vecteur de caractéristiques

        """

        all_features = []

        

        if not metrics_data:

            logger.warning("Aucune donnée à traiter")

            return pd.DataFrame()

        

        for service, service_metrics in metrics_data.items():

            for metric_type, df in service_metrics.items():

                if df.empty or len(df) < 5:

                    continue

                

                try:

                    feature_dict = {

                        "service": service,

                        "metric_type": metric_type,

                        "timestamp": datetime.now(),

                        "mean": float(df.mean().mean()),

                        "std": float(df.std().mean()),

                        "max": float(df.max().max()),

                        "min": float(df.min().min()),

                        "last_value": float(df.iloc[-1].mean()),

                        "trend": self._calculate_trend(df),

                        "data_points": len(df)

                    }

                    all_features.append(feature_dict)

                except Exception as e:

                    logger.error(f"Erreur pour {service}/{metric_type}: {e}")

                    continue

        

        if not all_features:

            return pd.DataFrame()

        

        features_df = pd.DataFrame(all_features)

        features_df = features_df.fillna(0)

        

        logger.info(f"Créé {len(features_df)} vecteurs de caractéristiques")

        return features_df

    

    def _calculate_trend(self, df, window=10):

        """

        Calcule la tendance des dernières valeurs

        """

        if len(df) < window:

            return 0.0

        

        try:

            if len(df.columns) > 1:

                last_values = df.iloc[-window:, 0].values

            else:

                last_values = df.iloc[-window:, 0].values

            

            last_values = last_values[~np.isnan(last_values)]

            

            if len(last_values) < 3:

                return 0.0

            

            x = np.arange(len(last_values))

            slope = np.polyfit(x, last_values, 1)[0]

            

            mean_val = np.mean(last_values)

            if mean_val > 0:

                slope_normalized = slope / mean_val

            else:

                slope_normalized = 0.0

            

            slope_normalized = max(-1.0, min(1.0, slope_normalized))

            return float(slope_normalized)

            

        except Exception as e:

            logger.error(f"Erreur calcul tendance: {e}")

            return 0.0

    

    def prepare_for_training(self, features_df):

        """

        Prépare les données pour l'entraînement

        """

        if features_df.empty:

            logger.warning("DataFrame vide")

            return None, features_df

        

        missing_cols = [col for col in self.feature_columns if col not in features_df.columns]

        if missing_cols:

            logger.error(f"Colonnes manquantes: {missing_cols}")

            return None, features_df

        

        X = features_df[self.feature_columns].values

        X = self.imputer.fit_transform(X)

        X_scaled = self.scaler.fit_transform(X)

        

        self.is_fitted = True

        logger.info(f"Données préparées: {X_scaled.shape}")

        return X_scaled, features_df

    

    def transform(self, features_df):

        """

        Transforme de nouvelles données

        """

        if not self.is_fitted:

            logger.error("Le préprocesseur n'est pas entraîné")

            return None, features_df

        

        if features_df.empty:

            return None, features_df

        

        X = features_df[self.feature_columns].values

        X = self.imputer.transform(X)

        X_scaled = self.scaler.transform(X)

        

        return X_scaled, features_df
