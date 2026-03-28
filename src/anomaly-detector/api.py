from flask import Flask, jsonify, request

from flask_cors import CORS

import threading

import time

import logging

import os

from datetime import datetime



from metrics_collector import MetricsCollector

from data_preprocessor import DataPreprocessor

from anomaly_detector import AnomalyDetector



app = Flask(__name__)

CORS(app)



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



# Configuration

PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'https://prometheus-k8s-openshift-monitoring.apps.sno.openshift.local')

PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')



# Initialisation

collector = MetricsCollector(prometheus_url=PROMETHEUS_URL, token=PROMETHEUS_TOKEN)

preprocessor = DataPreprocessor()

detector = AnomalyDetector(contamination=0.1)



# Stockage

current_anomalies = []

anomaly_history = []

last_detection_time = None



def train_initial_model():

    """Entraîne le modèle initial"""

    logger.info("🔄 Entraînement initial...")

    

    try:

        metrics_data = collector.collect_all_metrics(hours_back=24)

        if not metrics_data:

            logger.error("Aucune donnée")

            return False

        

        features_df = preprocessor.create_feature_vector(metrics_data)

        if features_df.empty:

            logger.error("Aucune feature")

            return False

        

        X_scaled, features_df = preprocessor.prepare_for_training(features_df)

        if X_scaled is None:

            return False

        

        success = detector.train(X_scaled, features_df)

        if success:

            detector.save_model()

            logger.info("✅ Entraînement terminé")

        return success

        

    except Exception as e:

        logger.error(f"Erreur: {e}")

        return False



def detection_loop(interval_minutes=5):

    """Boucle de détection"""

    global current_anomalies, anomaly_history, last_detection_time

    

    time.sleep(10)

    

    while True:

        try:

            logger.info("🔍 Détection d'anomalies...")

            last_detection_time = datetime.now()

            

            metrics_data = collector.collect_all_metrics(hours_back=1)

            if not metrics_data:

                time.sleep(interval_minutes * 60)

                continue

            

            features_df = preprocessor.create_feature_vector(metrics_data)

            if features_df.empty:

                time.sleep(interval_minutes * 60)

                continue

            

            X_scaled, features_df = preprocessor.transform(features_df)

            if X_scaled is None:

                time.sleep(interval_minutes * 60)

                continue

            

            anomalies = detector.analyze_current_data(X_scaled, features_df)

            current_anomalies = anomalies

            

            for anomaly in anomalies:

                if anomaly not in anomaly_history[-50:]:

                    anomaly_history.append(anomaly)

                    logger.warning(f"⚠️ {anomaly['service']} - {anomaly['metric_type']} - {anomaly['severity']}")

            

            while len(anomaly_history) > 100:

                anomaly_history.pop(0)

            

            logger.info(f"✅ Détection terminée - {len(anomalies)} anomalies")

            

        except Exception as e:

            logger.error(f"Erreur: {e}")

        

        time.sleep(interval_minutes * 60)



@app.route('/api/health', methods=['GET'])

def health():

    return jsonify({

        "status": "healthy",

        "timestamp": datetime.now().isoformat(),

        "last_detection": last_detection_time.isoformat() if last_detection_time else None,

        "anomalies_count": len(current_anomalies),

        "model_trained": detector.is_trained

    })



@app.route('/api/anomalies/current', methods=['GET'])

def get_anomalies():

    return jsonify({

        "timestamp": datetime.now().isoformat(),

        "total": len(current_anomalies),

        "anomalies": current_anomalies

    })



@app.route('/api/anomalies/history', methods=['GET'])

def get_history():

    limit = request.args.get('limit', 50, type=int)

    return jsonify({

        "total": len(anomaly_history),

        "anomalies": anomaly_history[-limit:]

    })



@app.route('/api/anomalies/dashboard', methods=['GET'])

def get_dashboard():

    severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    services = {}

    

    for a in current_anomalies:

        severity[a["severity"]] += 1

        services[a["service"]] = services.get(a["service"], 0) + 1

    

    return jsonify({

        "summary": {

            "total": len(current_anomalies),

            "critical": severity["CRITICAL"],

            "high": severity["HIGH"],

            "medium": severity["MEDIUM"],

            "low": severity["LOW"]

        },

        "severity": severity,

        "services": services,

        "anomalies": current_anomalies,

        "last_update": datetime.now().isoformat()

    })



@app.route('/api/status', methods=['GET'])

def get_status():

    return jsonify({

        "model_trained": detector.is_trained,

        "preprocessor_ready": preprocessor.is_fitted,

        "last_detection": last_detection_time.isoformat() if last_detection_time else None

    })



@app.route('/api/train', methods=['POST'])

def retrain():

    success = train_initial_model()

    return jsonify({"success": success, "message": "Entraînement terminé" if success else "Échec"})



if __name__ == '__main__':

    logger.info("🚀 Démarrage API...")
    
    # Essayer de charger le préprocesseur
    if not preprocessor.load():
        logger.info("Aucun préprocesseur existant, entraînement...")
        train_initial_model()
        preprocessor.save()  # Sauvegarder après entraînement
    else:
        logger.info("✅ Préprocesseur chargé")

    

    # Essayer de charger un modèle existant

    if not detector.load_model():

        logger.info("Aucun modèle existant, entraînement...")

        train_initial_model()
        detector.save_model()
    

    # Démarrer la détection

    thread = threading.Thread(target=detection_loop, args=(5,), daemon=True)

    thread.start()

    

    port = int(os.environ.get('PORT', 5000))

    app.run(host='0.0.0.0', port=port, debug=False)


