import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from typing import Tuple, Optional, List


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
        print("Calibration du modèle GARCH(1,1)...")
        self.model = arch_model(centered_returns, vol='GARCH', p=1, q=1, rescale=False)
        self.results = self.model.fit(disp='off')  # Désactiver l'affichage détaillé
        
        # Extraire les paramètres
        self.params = {
            'omega': self.results.params['omega'],
            'alpha': self.results.params['alpha[1]'],
            'beta': self.results.params['beta[1]'],
            'mean_return': self.mean_return
        }
        
        print("Paramètres GARCH calibrés:")
        print(f"  omega: {self.params['omega']:.8f}")
        print(f"  alpha: {self.params['alpha']:.8f}")
        print(f"  beta: {self.params['beta']:.8f}")
        print(f"  mean_return: {self.params['mean_return']:.8f}")
        
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


class GarchSimulator:
    """
    Simulateur de prix basé sur un modèle GARCH calibré
    """
    def __init__(self, params: dict, initial_price: float, initial_volatility: Optional[float] = None):
        """
        Initialiser le simulateur GARCH
        
        Args:
            params: Paramètres du modèle GARCH (omega, alpha, beta, mean_return)
            initial_price: Prix initial pour la simulation
            initial_volatility: Volatilité initiale (si None, utilisera la variance inconditionnelle)
        """
        self.omega = params['omega']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.mean_return = params['mean_return']
        self.price = initial_price
        
        # Calculer la variance inconditionnelle comme valeur initiale par défaut
        uncond_variance = self.omega / (1 - self.alpha - self.beta)
        
        # Utiliser la volatilité spécifiée ou la variance inconditionnelle
        self.last_variance = initial_volatility**2 if initial_volatility else uncond_variance
        self.last_return = 0.0

    def step(self) -> Tuple[float, float]:
        """
        Simuler un pas de temps selon le processus GARCH
        
        Returns:
            tuple: (nouveau prix, nouvelle volatilité)
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
        
        # Mettre à jour le prix
        self.price = self.price * np.exp(current_return)
        
        # Stocker pour le prochain pas
        self.last_variance = current_variance
        self.last_return = current_return - self.mean_return  # Stocker le rendement centré
        
        return self.price, current_volatility
    
    def simulate_path(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simuler une trajectoire de prix sur plusieurs pas de temps
        
        Args:
            n_steps: Nombre de pas de simulation
            
        Returns:
            tuple: (trajectoire des prix, trajectoire des volatilités)
        """
        prices = np.zeros(n_steps + 1)
        volatilities = np.zeros(n_steps + 1)
        
        # Valeurs initiales
        prices[0] = self.price
        volatilities[0] = np.sqrt(self.last_variance)
        
        # Simuler la trajectoire
        for i in range(n_steps):
            prices[i+1], volatilities[i+1] = self.step()
            
        return prices, volatilities
    
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


class MarketEnvWithGarch:
    """
    Environnement de marché utilisant un simulateur GARCH pour l'exécution d'ordres
    Cette classe est une ébauche qui peut être adaptée pour implémenter un environnement Gymnasium complet
    """
    def __init__(self, 
                 garch_params: dict, 
                 initial_price: float, 
                 initial_volatility: float,
                 impact_factor: float = 0.1,
                 max_quantity: float = 1.0,
                 max_steps: int = 60):
        """
        Initialiser l'environnement
        
        Args:
            garch_params: Paramètres du modèle GARCH
            initial_price: Prix initial
            initial_volatility: Volatilité initiale
            impact_factor: Facteur d'impact de marché (lambda)
            max_quantity: Quantité totale à exécuter
            max_steps: Nombre maximum de pas par épisode
        """
        # Initialiser le simulateur GARCH
        self.simulator = GarchSimulator(garch_params, initial_price, initial_volatility)
        
        # Paramètres d'exécution
        self.impact_factor = impact_factor
        self.max_quantity = max_quantity
        self.max_steps = max_steps
        
        # État de l'environnement
        self.current_step = 0
        self.remaining_quantity = max_quantity
        self.price_no_impact = initial_price
        self.volatility = initial_volatility
        self.cash = 0.0
        self.done = False

    def reset(self, 
              seed: Optional[int] = None, 
              initial_price: Optional[float] = None,
              initial_volatility: Optional[float] = None) -> List[float]:
        """
        Réinitialiser l'environnement au début d'un nouvel épisode
        
        Args:
            seed: Graine aléatoire
            initial_price: Prix initial (si None, valeur par défaut)
            initial_volatility: Volatilité initiale (si None, valeur par défaut)
            
        Returns:
            List: État initial normalisé
        """
        # Fixer la graine aléatoire si spécifiée
        if seed is not None:
            np.random.seed(seed)
            
        # Réinitialiser les variables d'état
        self.current_step = 0
        self.remaining_quantity = self.max_quantity
        self.cash = 0.0
        self.done = False
        
        # Réinitialiser le simulateur
        self.simulator.reset(initial_price, initial_volatility)
        
        # Obtenir le prix et la volatilité initiaux
        self.price_no_impact, self.volatility = self.simulator.step()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[List[float], float, bool, dict]:
        """
        Exécuter une action et avancer l'environnement d'un pas
        
        Args:
            action: Indice correspondant à la fraction de l'inventaire restant à vendre
            
        Returns:
            tuple: (nouvel état, récompense, terminé, info)
        """
        # Mapping de l'action à une quantité à vendre
        # Par exemple, action 0->0%, 1->10%, 2->20%... 10->100% de l'inventaire restant
        action_values = np.linspace(0, 1, 11)
        fraction = action_values[action]
        quantity_to_sell = fraction * self.remaining_quantity
        
        # Calculer l'impact de marché et le prix effectif
        relative_order_size = quantity_to_sell / self.max_quantity
        price_impact = self.impact_factor * np.sqrt(relative_order_size) * self.price_no_impact * (self.volatility / 0.2)
        execution_price = self.price_no_impact - price_impact
        
        # Mettre à jour le cash et la quantité restante
        cash_received = quantity_to_sell * execution_price
        self.cash += cash_received
        self.remaining_quantity -= quantity_to_sell
        
        # Simuler le prochain prix et la volatilité
        self.price_no_impact, self.volatility = self.simulator.step()
        
        # Incrémenter le compteur d'étapes
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        self.done = (self.current_step >= self.max_steps) or (self.remaining_quantity <= 1e-6)
        
        # Récompense = cash reçu lors de cette étape
        reward = cash_received
        
        # Informations additionnelles
        info = {
            "price": self.price_no_impact,
            "volatility": self.volatility,
            "cash": self.cash,
            "remaining": self.remaining_quantity,
            "step": self.current_step,
            "execution_price": execution_price,
            "price_impact": price_impact
        }
        
        return self._get_state(), reward, self.done, info
    
    def _get_state(self) -> List[float]:
        """
        Construire le vecteur d'état normalisé
        
        Returns:
            List: État normalisé (inventaire normalisé, temps normalisé, volatilité normalisée)
        """
        # Normalisation simple
        normalized_inventory = self.remaining_quantity / self.max_quantity
        normalized_time = self.current_step / self.max_steps
        normalized_volatility = self.volatility / 0.2  # Supposons 0.2 comme volatilité typique
        
        # État augmenté
        # Ici, on pourrait ajouter d'autres features comme MACD, position de la bougie, etc.
        
        return [normalized_inventory, normalized_time, normalized_volatility]


# Code d'utilisation
if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    
    # Charger les données (supposons qu'elles sont déjà disponibles)
    data = pd.read_csv('data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv', index_col=0, parse_dates=True)
    
    # Calibrer le modèle GARCH
    calibrator = GarchCalibrator()
    params = calibrator.fit(data['close'])
    
    # Visualiser les diagnostics
    calibrator.plot_diagnostics(data['close'])
    
    # Créer un simulateur avec les paramètres calibrés
    initial_price = data['close'].iloc[-1]
    initial_volatility = np.sqrt(calibrator.results.conditional_volatility.iloc[-1]**2)
    simulator = GarchSimulator(params, initial_price, initial_volatility)
    
    # Simuler une trajectoire et la visualiser
    n_steps = 1000
    prices, vols = simulator.simulate_path(n_steps)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(prices)
    plt.title('Prix simulés avec modèle GARCH(1,1)')
    plt.ylabel('Prix')
    
    plt.subplot(2, 1, 2)
    plt.plot(vols)
    plt.title('Volatilité simulée')
    plt.ylabel('Volatilité')
    plt.xlabel('Pas de temps')
    
    plt.tight_layout()
    plt.show()
    
    # Exemple d'utilisation de l'environnement
    print("\nCréation de l'environnement d'exécution avec GARCH...")
    env = MarketEnvWithGarch(params, initial_price, initial_volatility)
    
    # Simuler quelques actions
    state = env.reset()
    print(f"État initial: {state}")
    
    for i in range(10):
        # Action aléatoire (0 à 10)
        action = np.random.randint(0, 11)
        next_state, reward, done, info = env.step(action)
        
        print(f"\nÉtape {i+1}")
        print(f"  Action: {action}")
        print(f"  Récompense: {reward:.2f}")
        print(f"  Nouvel état: {next_state}")
        print(f"  Prix: {info['price']:.2f}, Volatilité: {info['volatility']:.4f}")
        print(f"  Inventaire restant: {info['remaining']:.4f} ({info['remaining']/env.max_quantity*100:.1f}%)")
        
        if done:
            print("\nÉpisode terminé!")
            break