import sys
import pandas as pd
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Classe pour récupérer les métriques depuis Prometheus
    """
    
    def __init__(self, prometheus_url=None, token=None):
        """
        Initialise le connecteur Prometheus
        
        Args:
            prometheus_url: URL du serveur Prometheus
            token: Token d'authentification Bearer
        """
        # Utiliser l'URL externe par défaut
        if prometheus_url is None:
            prometheus_url = "https://prometheus-k8s-openshift-monitoring.apps.sno.openshift.local"
        
        # Récupérer le token depuis l'environnement ou le paramètre
        if token is None:
            token = os.environ.get('PROMETHEUS_TOKEN', '')
        
        logger.info(f"Connexion à Prometheus: {prometheus_url}")
        
        # Configuration pour l'authentification
        headers = None
        if token:
            headers = {"Authorization": f"Bearer {token}"}
        
        self.prometheus = PrometheusConnect(
            url=prometheus_url,
            headers=headers,
            disable_ssl=True  # Désactiver SSL car certificat auto-signé
        )
        
    def get_metrics_for_service(self, service_name, metric_type, hours_back=24):
        """
        Récupère les métriques pour un service spécifique
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Construire la requête PromQL selon le type de métrique
        if metric_type == "cpu":
            query = f'container_cpu_usage_seconds_total{{namespace="online-boutique-dev", pod=~"{service_name}.*"}}'
        elif metric_type == "memory":
            query = f'container_memory_working_set_bytes{{namespace="online-boutique-dev", pod=~"{service_name}.*"}}'
        elif metric_type == "up":
            query = f'up{{namespace="online-boutique-dev", pod=~"{service_name}.*"}}'
        else:
            logger.warning(f"Type de métrique inconnu: {metric_type}")
            return pd.DataFrame()
        
        logger.debug(f"Requête PromQL: {query}")
        
        try:
            # Récupérer les données
            result = self.prometheus.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step="1m"
            )
            
            # Convertir en DataFrame
            data = []
            for item in result:
                if "values" in item:
                    for timestamp, value in item["values"]:
                        try:
                            data.append({
                                "timestamp": datetime.fromtimestamp(timestamp),
                                "value": float(value),
                                "pod": item["metric"].get("pod", service_name),
                                "metric": metric_type
                            })
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Erreur conversion valeur: {value} - {e}")
                            continue
            
            if not data:
                logger.warning(f"Aucune donnée pour {service_name} - {metric_type}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Pivoter pour avoir une colonne par pod
            if not df.empty and 'timestamp' in df.columns:
                df_pivot = df.pivot_table(
                    index="timestamp", 
                    columns="pod", 
                    values="value"
                )
                # Remplir les valeurs manquantes
                df_pivot = df_pivot.fillna(method="ffill").fillna(0)
                logger.info(f"Récupéré {len(df_pivot)} points pour {service_name} - {metric_type}")
                return df_pivot
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques pour {service_name}: {e}")
            return pd.DataFrame()
    
    def collect_all_metrics(self, services=None, hours_back=24):
        """
        Collecte toutes les métriques pour tous les services
        """
        if services is None:
            services = [
                "frontend", "cartservice", "productcatalogservice", 
                "checkoutservice", "shippingservice", "paymentservice",
                "emailservice", "currencyservice", "recommendationservice"
            ]
        
        all_metrics = {}
        
        logger.info(f"Début de la collecte pour {len(services)} services...")
        
        for service in services:
            logger.info(f"Collecte des métriques pour {service}...")
            service_metrics = {}
            
            # Collecter chaque type de métrique
            for metric_type in ["cpu", "memory", "up"]:
                try:
                    df = self.get_metrics_for_service(service, metric_type, hours_back)
                    if not df.empty:
                        service_metrics[metric_type] = df
                except Exception as e:
                    logger.error(f"Erreur pour {service}/{metric_type}: {e}")
            
            if service_metrics:
                all_metrics[service] = service_metrics
            else:
                logger.warning(f"Aucune donnée pour {service}")
        
        logger.info(f"Collecte terminée. {len(all_metrics)} services avec données")
        return all_metrics


if __name__ == "__main__":
    print("Test du collecteur de métriques...")
    
    # Récupérer le token
    token = os.environ.get('PROMETHEUS_TOKEN')
    
    if not token:
        print("⚠️  Token Prometheus non trouvé dans l'environnement")
        print("   Exportez votre token avec:")
        print("   export PROMETHEUS_TOKEN=$(oc whoami -t)")
        print("   ou")
        print("   export PROMETHEUS_TOKEN=$(oc create token prometheus-k8s -n openshift-monitoring)")
        sys.exit(1)
    
    collector = MetricsCollector(token=token)
    
    # Tester avec un seul service et 1 heure
    metrics = collector.collect_all_metrics(hours_back=1)
    
    for service, data in metrics.items():
        print(f"\n{service}:")
        for metric_type, df in data.items():
            if not df.empty:
                print(f"  {metric_type}: {df.shape} points")
                print(f"  Dernière valeur: {df.iloc[-1].values[0]:.2f}")
