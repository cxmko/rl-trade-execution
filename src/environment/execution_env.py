import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.environment.garch_simulator import GarchCalibrator, GarchSimulator, calibrate_full_model


class OptimalExecutionEnv(gym.Env):
    """
    Environnement d'exécution optimale compatible Gymnasium
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data_path: str,
        initial_inventory: float = 1000,
        horizon_steps: int = 60,
        lambda_0: float = 0.0005,
        alpha: float = 0.5,
        delta: float = 0.05,
        calibration_window: int = 5000,
        vol_window: int = 20,
        random_start_prob: float = 0.0,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Paramètres d'environnement
        self.initial_inventory = initial_inventory
        self.horizon_steps = horizon_steps
        self.vol_window = vol_window
        self.calibration_window = calibration_window
        self.random_start_prob = random_start_prob
        
        # Paramètres d'impact
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.delta = delta
        
        # Actions discrètes : [0, 0.5, 1, 2, 3, 5, 10, 25, 50, 75, 100] %
        self.action_percentages = np.array([0, 0.5, 1, 2, 3, 5, 10, 25, 50, 75, 100])
        self.action_space = spaces.Discrete(len(self.action_percentages))
        
        # État : [inventaire_norm, temps_restant_norm, vol_realized, prix_relatif]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.5]),
            high=np.array([1.0, 1.0, 1.0, 2.0]),
            dtype=np.float32
        )
        
        # Charger les données historiques
        print(f"Chargement des données depuis {data_path}...")
        self.historical_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Calibrer GARCH et modèle volume UNE SEULE FOIS (sur toutes les données)
        self._calibrate_global()
        
        # Variables d'état
        self.current_step = 0
        self.inventory = initial_inventory
        self.initial_price = 0.0
        self.prices_history = []
        self.volumes_history = []
        self.realized_vols_history = []
        self.garch_simulator = None
        self.total_revenue = 0.0
        self.random_start_idx = 0
        
    def _calibrate_global(self):
        """Calibrer les modèles GARCH et volume sur TOUTES les données (une seule fois)"""
        print("Calibration globale des modèles GARCH et volume...")
        self.garch_params, self.volume_params = calibrate_full_model(self.historical_data)
        print("✓ Calibration globale terminée")
        
    def _calculate_realized_volatility(self, prices: np.ndarray) -> float:
        """Calculer la volatilité réalisée sur les derniers prix"""
        if len(prices) < 2:
            return 0.001
        
        returns = np.diff(np.log(prices[-min(self.vol_window+1, len(prices)):]))
        
        if len(returns) < 2:
            return 0.001
            
        return np.std(returns)
    
    def _calculate_rolling_mean(self, history: list, window: int) -> float:
        """Calculer la moyenne mobile sur une fenêtre"""
        if len(history) < 2:
            return history[-1] if history else 1.0
        
        window_size = min(window, len(history))
        return np.mean(history[-window_size:])
    
    def _action_to_quantity(self, action: int) -> float:
        """Convertir l'action discrète en quantité à vendre"""
        percentage = self.action_percentages[action] / 100.0
        return self.inventory * percentage
    
    def _calculate_temporary_impact(self, quantity: float, realized_vol: float, 
                                    rolling_sigma: float, rolling_volume: float) -> float:
        """Calculer l'impact temporaire RELATIF (en pourcentage)"""
        if quantity <= 0:
            return 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # FIX : Ajouter des planchers robustes pour éviter l'explosion
        # ═══════════════════════════════════════════════════════════════
        MIN_ROLLING_VOL = 3.0  # Basé sur le P5.0 des données réelles
        MIN_ROLLING_SIGMA = 0.0001  # Un plancher raisonnable
        
        rolling_sigma = max(rolling_sigma, MIN_ROLLING_SIGMA)
        rolling_volume = max(rolling_volume, MIN_ROLLING_VOL)
        # ═══════════════════════════════════════════════════════════════
        
        vol_factor = 1 + (realized_vol / rolling_sigma)
        quantity_factor = (quantity / rolling_volume) ** self.alpha
        
        return self.lambda_0 * vol_factor * quantity_factor
    
    def _calculate_permanent_impact(self, temporary_impact_relative: float) -> float:
        """Calculer l'impact permanent RELATIF"""
        return self.delta * temporary_impact_relative
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Réinitialiser l'environnement avec possibilité de départ aléatoire"""
        super().reset(seed=seed)
        
        # ═══════════════════════════════════════════════════════════════
        # Gestion des départs aléatoires avec INVENTAIRE ALÉATOIRE
        # ═══════════════════════════════════════════════════════════════
        
        min_start = self.calibration_window + 1000
        max_start = len(self.historical_data) - self.horizon_steps - 100
        
        if max_start < min_start:
            raise ValueError(f"Données insuffisantes. Besoin d'au moins {min_start + self.horizon_steps + 100} lignes.")
        
        # Décider si on fait un départ aléatoire
        use_random_start = np.random.random() < self.random_start_prob
        
        if use_random_start:
            # 1. Départ aléatoire dans l'épisode (entre step 0 et horizon_steps-1)
            self.random_start_idx = np.random.randint(min_start, max_start)
            self.current_step = np.random.randint(0, self.horizon_steps)
            
            # 2. Commencer avec un inventaire aléatoire (proportionnel au temps, avec du bruit)
            #    Si 30% du temps est passé, on a vendu "environ" 30% du stock (±20%)
            random_time_elapsed_ratio = self.current_step / self.horizon_steps
            inventory_sold_ratio = random_time_elapsed_ratio * np.random.uniform(0.8, 1.2)
            inventory_sold_ratio = np.clip(inventory_sold_ratio, 0.0, 1.0)
            
            self.inventory = self.initial_inventory * (1.0 - inventory_sold_ratio)
        else:
            # Départ normal (début d'épisode, inventaire complet)
            self.random_start_idx = np.random.randint(min_start, max_start)
            self.current_step = 0
            self.inventory = self.initial_inventory
        
        # ═══════════════════════════════════════════════════════════════
        
        # Récupérer le prix RÉEL à ce point de départ
        self.initial_price = float(self.historical_data['close'].iloc[self.random_start_idx])
        
        # Calibrer GARCH localement
        window_start = self.random_start_idx - self.calibration_window
        window_data = self.historical_data.iloc[window_start:self.random_start_idx]
        
        calibrator = GarchCalibrator()
        local_params = calibrator.fit(window_data['close'])
        
        # Vérifier la stabilité du modèle local
        if local_params['alpha'] + local_params['beta'] >= 0.999:
            local_params = self.garch_params
        
        # Volatilité initiale
        initial_vol = np.sqrt(calibrator.results.conditional_volatility.iloc[-1]**2)
        initial_vol = np.clip(initial_vol, 0.0001, 0.05)
        
        # Initialiser le simulateur
        self.garch_simulator = GarchSimulator(
            local_params,
            self.initial_price,
            initial_vol,
            self.volume_params
        )
        
        # Réinitialiser l'état
        self.total_revenue = 0.0
        self.prices_history = [self.initial_price]
        
        # Warmup avec données réelles
        warmup_window = min(self.vol_window, self.random_start_idx - window_start)
        warmup_prices = self.historical_data['close'].iloc[self.random_start_idx-warmup_window:self.random_start_idx].values
        warmup_volumes = self.historical_data['volume'].iloc[self.random_start_idx-warmup_window:self.random_start_idx].values
        
        self.realized_vols_history = []
        for i in range(1, len(warmup_prices)):
            vol = self._calculate_realized_volatility(warmup_prices[:i+1])
            self.realized_vols_history.append(vol)
        
        self.volumes_history = list(warmup_volumes)
        
        state = self._get_state()
        
        return state, {}
    
    def _get_state(self) -> np.ndarray:
        """Obtenir l'état actuel"""
        inventory_norm = self.inventory / self.initial_inventory
        time_remaining_norm = max(0.0, (self.horizon_steps - self.current_step) / self.horizon_steps)
        realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        current_price = self.prices_history[-1]
        price_ratio = current_price / self.initial_price
        
        return np.array([
            inventory_norm,
            time_remaining_norm,
            realized_vol,
            price_ratio
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Exécuter une action - L'ÉPISODE DURE TOUJOURS 60 PAS"""
        time_remaining = self.horizon_steps - self.current_step
        forced_action = False
        
        # ═══════════════════════════════════════════════════════════════
        # FORCER LA VENTE AU DERNIER PAS DE TEMPS
        # ═══════════════════════════════════════════════════════════════
        if self.current_step >= self.horizon_steps - 1:
            quantity = self.inventory
            action_percentage = 1.0
            forced_action = True
        else:
            action_percentage = self.action_percentages[action] / 100.0
            quantity = self.inventory * action_percentage
            quantity = min(quantity, self.inventory)
        # ═══════════════════════════════════════════════════════════════
        
        current_price = self.prices_history[-1]
        realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        
        # Calculer normalisateurs
        rolling_sigma = self._calculate_rolling_mean(self.realized_vols_history, self.vol_window)
        rolling_volume = self._calculate_rolling_mean(self.volumes_history, self.vol_window)
        
        # Impact temporaire
        temp_impact_relative = self._calculate_temporary_impact(
            quantity, realized_vol, rolling_sigma, rolling_volume
        )
        
        execution_price = current_price * (1 - temp_impact_relative)
        
        # ═══════════════════════════════════════════════════════════════
        # FONCTION DE RÉCOMPENSE SIMPLE
        # ═══════════════════════════════════════════════════════════════
        
        revenue_raw = quantity * execution_price
        revenue_normalized = revenue_raw / self.initial_price
        
        # Récompense = revenu normalisé
        reward = revenue_normalized
        
        # ═══════════════════════════════════════════════════════════════
        
        self.total_revenue += revenue_raw
        self.inventory -= quantity
        
        # Impact permanent
        perm_impact_relative = self._calculate_permanent_impact(temp_impact_relative)
        perm_impact_relative = np.clip(perm_impact_relative, 0, 0.005)
        
        # Simuler le prochain prix
        next_price, next_vol, next_volume = self.garch_simulator.step()
        
        # Appliquer impact permanent avec AMORTISSEMENT
        impact_decay = 0.5
        next_price = next_price * (1 - perm_impact_relative * impact_decay)
        
        # Vérifier que le prix reste dans une plage raisonnable
        if next_price > self.initial_price * 2 or next_price < self.initial_price * 0.5:
            next_price = current_price * np.random.uniform(0.99, 1.01)
        
        # Ajouter à l'historique
        self.prices_history.append(next_price)
        self.volumes_history.append(next_volume)
        
        new_realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        self.realized_vols_history.append(new_realized_vol)
        
        self.current_step += 1
        
        # ═══════════════════════════════════════════════════════════════
        # FIX CRITIQUE : L'épisode se termine UNIQUEMENT après 60 pas
        # (Inventaire vide n'est PLUS une condition de terminaison)
        # ═══════════════════════════════════════════════════════════════
        terminated = (self.current_step >= self.horizon_steps)
        truncated = False
        # ═══════════════════════════════════════════════════════════════
        
        next_state = self._get_state()
        
        info = {
            'quantity_sold': quantity,
            'execution_price': execution_price,
            'temp_impact_relative': temp_impact_relative,
            'temp_impact_absolute': temp_impact_relative * current_price,
            'perm_impact_relative': perm_impact_relative,
            'perm_impact_absolute': perm_impact_relative * next_price,
            'inventory_remaining': self.inventory,
            'total_revenue': self.total_revenue,
            'current_price': next_price,
            'simulated_volume': next_volume,
            'rolling_sigma': rolling_sigma,
            'rolling_volume': rolling_volume,
            'forced_action': forced_action,
            'revenue_normalized': revenue_normalized,
            'net_reward': reward
        }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        """Afficher l'état actuel"""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}/{self.horizon_steps}")
            print(f"Inventory: {self.inventory:.4f}/{self.initial_inventory}")
            print(f"Current Price: {self.prices_history[-1]:.2f}")
            print(f"Total Revenue: {self.total_revenue:.2f}")
            print("-" * 50)



