import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.linear_model import Ridge
from typing import Tuple, Optional, Dict


class GarchCalibrator:
    """
    Classe pour calibrer un modèle GARCH(1,1) sur des données historiques
    """
    def __init__(self):
        self.model = None
        self.results = None
        self.params = None
        self.mean_return = None
        self.last_price = None
        self.last_volatility = None

    def fit(self, prices: pd.Series) -> dict:
        """
        Calibrer le modèle GARCH sur les données historiques
        
        Args:
            prices: Série de prix (close prices)
            
        Returns:
            dict: Paramètres calibrés du modèle GARCH
        """
        # Calculer les rendements logarithmiques
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Sauvegarder la moyenne des rendements
        self.mean_return = returns.mean()
        
        # Centrer les rendements (important pour la modélisation GARCH)
        centered_returns = returns - self.mean_return
        
        # Ajuster le modèle GARCH(1,1) aux rendements centrés
        self.model = arch_model(centered_returns, vol='GARCH', p=1, q=1, rescale=False)
        self.results = self.model.fit(disp='off', show_warning=False)  # ← AJOUTÉ show_warning=False
        
        # Extraire les paramètres
        self.params = {
            'omega': self.results.params['omega'],
            'alpha': self.results.params['alpha[1]'],
            'beta': self.results.params['beta[1]'],
            'mean_return': self.mean_return
        }
        
        # MODIFIÉ: Commenter tous les prints pour réduire spam
        # print("Paramètres GARCH calibrés:")
        # print(f"  omega: {self.params['omega']:.8f}")
        # print(f"  alpha: {self.params['alpha']:.8f}")
        # print(f"  beta: {self.params['beta']:.8f}")
        # print(f"  mean_return: {self.params['mean_return']:.8f}")
        
        # Sauvegarder le dernier prix et la dernière volatilité conditionnelle
        self.last_price = prices.iloc[-1]
        self.last_volatility = np.sqrt(self.results.conditional_volatility.iloc[-1]**2)
        
        # Vérifier la stabilité du modèle
        if self.params['alpha'] + self.params['beta'] >= 1:
            print("AVERTISSEMENT: Le modèle GARCH n'est pas stable (alpha + beta >= 1)")
        
        return self.params

    def plot_diagnostics(self, prices: pd.Series) -> None:
        """
        Tracer les graphiques de diagnostic du modèle GARCH
        
        Args:
            prices: Série de prix utilisée pour la calibration
        """
        if self.results is None:
            print("Vous devez d'abord calibrer le modèle avec la méthode fit()")
            return
        
        # Créer une figure avec plusieurs sous-graphiques
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. Prix et volatilité conditionnelle
        returns = np.log(prices / prices.shift(1)).dropna()
        vol = self.results.conditional_volatility
        
        ax1 = axs[0]
        ax1.plot(prices.index, prices.values, 'b-', label='Prix')
        ax1.set_title('Prix et Volatilité')
        ax1.set_ylabel('Prix')
        ax1.legend(loc='upper left')
        
        ax1b = ax1.twinx()
        ax1b.plot(vol.index, vol.values, 'r-', alpha=0.5, label='Volatilité conditionnelle')
        ax1b.set_ylabel('Volatilité')
        ax1b.legend(loc='upper right')
        
        # 2. Rendements et volatilité
        ax2 = axs[1]
        ax2.plot(returns.index, returns.values, 'b-', label='Rendements')
        ax2.set_title('Rendements et Volatilité conditionnelle')
        ax2.set_ylabel('Rendements')
        ax2.legend(loc='upper left')
        
        ax2b = ax2.twinx()
        ax2b.plot(vol.index, vol.values, 'r-', alpha=0.5, label='Volatilité conditionnelle')
        ax2b.set_ylabel('Volatilité')
        ax2b.legend(loc='upper right')
        
        # 3. QQ-plot des résidus standardisés
        from scipy import stats
        ax3 = axs[2]
        residuals = self.results.resid / self.results.conditional_volatility
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('QQ-Plot des résidus standardisés')
        
        plt.tight_layout()
        plt.show()


class VolumeModelCalibrator:
    """
    Classe pour calibrer un modèle volume-volatilité bimodal
    """
    def __init__(self):
        self.volume_params = None
        
    def calibrate(self, volumes: np.ndarray, conditional_variances: np.ndarray) -> Dict:
        """
        Calibrer le modèle volume-volatilité avec approche bimodale
        
        Args:
            volumes: Array des volumes observés
            conditional_variances: Array des variances conditionnelles GARCH
            
        Returns:
            dict: Paramètres du modèle volume
        """
        print("\n=== Calibration du modèle Volume-Volatilité (2 régimes) ===")
        
        # Identifier les deux régimes basés sur un seuil
        volume_threshold = np.percentile(volumes, 25)
        low_volume_mask = volumes <= volume_threshold
        high_volume_mask = volumes > volume_threshold
        
        print(f"Seuil volume: {volume_threshold:.2f}")
        print(f"Proportion faible volume: {low_volume_mask.sum() / len(volumes) * 100:.1f}%")
        print(f"Proportion fort volume: {high_volume_mask.sum() / len(volumes) * 100:.1f}%")
        
        # Calculer le 99ème percentile et le max pour le clipping
        volume_99p = np.percentile(volumes, 99)
        volume_max = np.max(volumes)
        
        print(f"Volume 99ème percentile: {volume_99p:.2f}")
        print(f"Volume maximum observé: {volume_max:.2f}")
        
        # --- RÉGIME FAIBLE VOLUME ---
        low_volumes = volumes[low_volume_mask]
        low_volume_mean = np.mean(low_volumes)
        low_volume_std = np.std(low_volumes)
        
        print(f"\nRégime faible volume:")
        print(f"  Moyenne: {low_volume_mean:.2f}")
        print(f"  Écart-type: {low_volume_std:.2f}")
        
        # --- RÉGIME FORT VOLUME ---
        high_volumes = volumes[high_volume_mask]
        high_variances = conditional_variances[high_volume_mask]
        
        # Régression log-log
        log_high_volumes = np.log(high_volumes + 1e-10)
        log_high_variances = np.log(high_variances + 1e-10)
        
        model_high = Ridge(alpha=1.0)
        model_high.fit(log_high_variances.reshape(-1, 1), log_high_volumes)
        
        c0_high = model_high.intercept_
        c1_high = model_high.coef_[0]
        
        predictions_high = model_high.predict(log_high_variances.reshape(-1, 1))
        residuals_high = log_high_volumes - predictions_high
        high_volume_noise_std = np.std(residuals_high) * 0.7
        
        print(f"\nRégime fort volume:")
        print(f"  c0 (intercept): {c0_high:.4f}")
        print(f"  c1 (slope): {c1_high:.4f}")
        print(f"  Écart-type du bruit: {high_volume_noise_std:.4f}")
        print(f"  R²: {model_high.score(log_high_variances.reshape(-1, 1), log_high_volumes):.4f}")
        
        self.volume_params = {
            'volume_threshold': volume_threshold,
            'low_volume_mean': low_volume_mean,
            'low_volume_std': low_volume_std,
            'c0_high': c0_high,
            'c1_high': c1_high,
            'high_volume_noise_std': high_volume_noise_std,
            'prob_low_volume': low_volume_mask.sum() / len(volumes),
            'volume_99p': volume_99p,
            'volume_max_observed': volume_max
        }
        
        return self.volume_params


class GarchSimulator:
    """
    Simulateur de prix et volumes basé sur un modèle GARCH calibré
    """
    def __init__(self, 
                 garch_params: dict, 
                 initial_price: float, 
                 initial_volatility: Optional[float] = None,
                 volume_params: Optional[Dict] = None):
        """
        Initialiser le simulateur GARCH
        
        Args:
            garch_params: Paramètres du modèle GARCH (omega, alpha, beta, mean_return)
            initial_price: Prix initial pour la simulation
            initial_volatility: Volatilité initiale (si None, utilisera la variance inconditionnelle)
            volume_params: Paramètres du modèle volume (optionnel)
        """
        self.omega = garch_params['omega']
        self.alpha = garch_params['alpha']
        self.beta = garch_params['beta']
        self.mean_return = garch_params['mean_return']
        self.price = initial_price
        
        # Paramètres volume (optionnel)
        self.volume_params = volume_params
        
        # Calculer la variance inconditionnelle comme valeur initiale par défaut
        uncond_variance = self.omega / (1 - self.alpha - self.beta)
        
        # Utiliser la volatilité spécifiée ou la variance inconditionnelle
        self.last_variance = initial_volatility**2 if initial_volatility else uncond_variance
        self.last_return = 0.0

    def _generate_next_return(self) -> float:
        """
        Générer le prochain rendement selon le processus GARCH (méthode interne)
        
        Returns:
            float: Le rendement généré
        """
        # Mettre à jour la variance conditionnelle selon le modèle GARCH(1,1)
        current_variance = (self.omega + 
                           self.alpha * self.last_return**2 + 
                           self.beta * self.last_variance)
        
        # Générer un choc aléatoire
        z = np.random.normal(0, 1)
        
        # Calculer le rendement selon le modèle
        current_volatility = np.sqrt(current_variance)
        current_return = self.mean_return + current_volatility * z
        
        # Stocker pour le prochain pas
        self.last_variance = current_variance
        self.last_return = current_return - self.mean_return
        
        return current_return
    
    def _generate_volume(self) -> float:
        """
        Générer un volume selon le modèle bimodal
        
        Returns:
            float: Volume généré
        """
        if self.volume_params is None:
            return 0.0
        
        variance = self.last_variance
        
        # Tirer au sort le régime
        if np.random.random() < self.volume_params['prob_low_volume']:
            # RÉGIME FAIBLE VOLUME
            volume = np.abs(np.random.normal(
                self.volume_params['low_volume_mean'], 
                self.volume_params['low_volume_std']
            ))
            volume = min(volume, self.volume_params['volume_threshold'])
        else:
            # RÉGIME FORT VOLUME
            log_variance = np.log(variance + 1e-10)
            noise = np.random.normal(0, self.volume_params['high_volume_noise_std'])
            log_volume = self.volume_params['c0_high'] + self.volume_params['c1_high'] * log_variance + noise
            volume = np.exp(log_volume)
            
            volume = max(volume, self.volume_params['volume_threshold'])
            
            # Si dépasse le max observé, tirer aléatoirement entre p99 et max
            if volume > self.volume_params['volume_max_observed']:
                volume = np.random.uniform(
                    self.volume_params['volume_99p'],
                    self.volume_params['volume_max_observed']
                )
        
        return volume

    def step(self) -> Tuple[float, float, float]:
        """
        Simuler un pas de temps selon le processus GARCH
        
        Returns:
            tuple: (nouveau prix, nouvelle volatilité, nouveau volume)
        """
        current_return = self._generate_next_return()
        
        # Mettre à jour le prix
        self.price = self.price * np.exp(current_return)
        
        current_volatility = np.sqrt(self.last_variance)
        
        # Générer le volume si le modèle est disponible
        current_volume = self._generate_volume()
        
        return self.price, current_volatility, current_volume
    
    def simulate_path(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simuler une trajectoire de prix sur plusieurs pas de temps
        
        Args:
            n_steps: Nombre de pas de simulation
            
        Returns:
            tuple: (trajectoire des prix, trajectoire des volatilités, trajectoire des volumes)
        """
        prices = np.zeros(n_steps + 1)
        volatilities = np.zeros(n_steps + 1)
        volumes = np.zeros(n_steps) if self.volume_params else None
        
        # Valeurs initiales
        prices[0] = self.price
        volatilities[0] = np.sqrt(self.last_variance)
        
        # Simuler la trajectoire
        for i in range(n_steps):
            prices[i+1], volatilities[i+1], volume = self.step()
            if volumes is not None:
                volumes[i] = volume
            
        return prices, volatilities, volumes
    
    def reset(self, price: Optional[float] = None, volatility: Optional[float] = None) -> None:
        """
        Réinitialiser le simulateur avec de nouvelles valeurs initiales
        
        Args:
            price: Nouveau prix initial (si None, garde la valeur actuelle)
            volatility: Nouvelle volatilité initiale (si None, garde la valeur actuelle)
        """
        if price is not None:
            self.price = price
            
        if volatility is not None:
            self.last_variance = volatility**2
            
        self.last_return = 0.0


def calibrate_full_model(data: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Calibrer à la fois le modèle GARCH et le modèle volume
    
    Args:
        data: DataFrame avec colonnes 'close' et 'volume'
        
    Returns:
        tuple: (garch_params, volume_params)
    """
    # Calibrer GARCH
    garch_calibrator = GarchCalibrator()
    garch_params = garch_calibrator.fit(data['close'])
    
    # Calibrer modèle volume
    conditional_variances = garch_calibrator.results.conditional_volatility ** 2
    volumes = data['volume'].values[len(data) - len(conditional_variances):]
    
    volume_calibrator = VolumeModelCalibrator()
    volume_params = volume_calibrator.calibrate(volumes, conditional_variances.values)
    
    return garch_params, volume_params


# Code d'exemple
if __name__ == "__main__":
    import pandas as pd
    
    # Charger les données
    data = pd.read_csv('../../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv', 
                       index_col=0, parse_dates=True)
    
    # Calibrer les modèles
    garch_params, volume_params = calibrate_full_model(data)
    
    # Créer un simulateur
    initial_price = data['close'].iloc[-1]
    initial_volatility = np.sqrt((garch_params['omega'] / 
                                 (1 - garch_params['alpha'] - garch_params['beta'])))
    
    simulator = GarchSimulator(garch_params, initial_price, initial_volatility, volume_params)
    
    # Simuler une trajectoire
    prices, vols, volumes = simulator.simulate_path(1000)
    
    # Visualiser
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    axs[0].plot(prices)
    axs[0].set_title('Prix simulés (GARCH)')
    axs[0].set_ylabel('Prix')
    
    axs[1].plot(vols)
    axs[1].set_title('Volatilité simulée')
    axs[1].set_ylabel('Volatilité')
    
    axs[2].plot(volumes)
    axs[2].set_title('Volumes simulés (Modèle bimodal)')
    axs[2].set_ylabel('Volume')
    axs[2].set_xlabel('Pas de temps')
    
    plt.tight_layout()
    plt.show()