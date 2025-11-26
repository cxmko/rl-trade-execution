import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.environment.garch_simulator import GarchCalibrator, GarchSimulator, calibrate_full_model


class OptimalExecutionEnv(gym.Env):
    """
    Environnement d'exécution optimale compatible Gymnasium
    Récompense : Implementation Shortfall (IS) vs Arrival Price
    """
    metadata = {'render_modes': ['human']}
    
    # ✅ CONSTANTS: Safety thresholds for numerical stability
    MIN_ROLLING_VOL = 0.01       
    MIN_ROLLING_SIGMA = 1e-6     
    MAX_IMPACT_PCT = 0.05        
    
    def __init__(
        self,
        data_path: str,
        initial_inventory: float = 1000,
        horizon_steps: int = 240,  # ✅ CHANGED: 60 -> 240 (4 hours)
        lambda_0: float = 0.004,
        alpha: float = 0.5,
        delta: float = 0.05,
        calibration_window: int = 5000,
        vol_window: int = 5,    
        avg_window: int = 60,   
        random_start_prob: float = 0.0,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Paramètres d'environnement
        self.initial_inventory = initial_inventory
        self.horizon_steps = horizon_steps
        self.vol_window = vol_window
        self.avg_window = avg_window
        self.calibration_window = calibration_window
        self.random_start_prob = random_start_prob
        
        # Paramètres d'impact
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.delta = delta
        
        # ✅ CHANGED: Action Space
        # Actions are now % of REMAINING inventory.
        # 0=0%, 1=10%, ..., 10=100% (Liquidation)
        # This guarantees the agent can always clear the position.
        self.action_percentages = np.array([0, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 75, 100])
        self.action_space = spaces.Discrete(len(self.action_percentages))
        
        # ✅ CHANGED: State Space
        # Removed Sharpe Ratio. Dim = 5.
        # [inventory_norm, time_remaining_norm, vol_ratio, liquidity_ratio, price_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.5]),  
            high=np.array([1.0, 1.0, 10.0, 100.0, 2.0]), 
            dtype=np.float32
        )
        
        # Charger les données historiques
        print(f"Chargement des données depuis {data_path}...")
        self.historical_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Calibrer GARCH et modèle volume UNE SEULE FOIS
        self._calibrate_global()
        
        # Variables d'état
        self.current_step = 0
        self.inventory = initial_inventory
        self.twap_inventory = 0.0
        self.initial_price = 0.0
        self.prices_history = []
        self.volumes_history = []
        self.realized_vols_history = []
        self.garch_simulator = None
        self.total_revenue = 0.0
        self.random_start_idx = 0
        self.inventory_rand = 0
    
    def _calibrate_global(self):
        """Calibrer les modèles GARCH et volume sur TOUTES les données"""
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
    
    def _calculate_temporary_impact(self, quantity: float, realized_vol: float, 
                                    rolling_sigma: float, rolling_volume: float) -> float:
        """Calculer l'impact temporaire RELATIF (en pourcentage)"""
        if quantity <= 0:
            return 0.0
        
        rolling_sigma = max(rolling_sigma, self.MIN_ROLLING_SIGMA)
        rolling_volume = max(rolling_volume, self.MIN_ROLLING_VOL)
        
        vol_factor = 1 + (realized_vol / rolling_sigma)
        quantity_factor = (quantity / rolling_volume) ** self.alpha
        
        raw_impact = self.lambda_0 * vol_factor * quantity_factor
        
        clipped_impact = min(raw_impact, self.MAX_IMPACT_PCT)
        
        return clipped_impact
    
    def _calculate_permanent_impact(self, temporary_impact_relative: float) -> float:
        """Calculer l'impact permanent RELATIF"""
        return self.delta * temporary_impact_relative
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Réinitialiser l'environnement avec possibilité de départ aléatoire"""
        super().reset(seed=seed)
        
        # Gestion des départs aléatoires
        min_start = self.calibration_window + 1000
        max_start = len(self.historical_data) - self.horizon_steps - 100
        
        if max_start < min_start:
            raise ValueError(f"Données insuffisantes.")
        
        use_random_start = np.random.random() < self.random_start_prob
        
        if use_random_start:
            self.random_start_idx = np.random.randint(min_start, max_start)
            self.current_step = np.random.randint(0, self.horizon_steps)
            
            random_time_elapsed_ratio = self.current_step / self.horizon_steps
            inventory_sold_ratio = random_time_elapsed_ratio * np.random.uniform(0.8, 1.2)
            inventory_sold_ratio = np.clip(inventory_sold_ratio, 0.0, 1.0)
            
            self.inventory = self.initial_inventory * (1.0 - inventory_sold_ratio)
            self.inventory_rand = self.inventory
        else:
            self.random_start_idx = np.random.randint(min_start, max_start)
            self.current_step = 0
            self.inventory = self.initial_inventory
            self.inventory_rand = self.initial_inventory
        
        self.twap_inventory = self.inventory_rand
        
        self.initial_price = float(self.historical_data['close'].iloc[self.random_start_idx])
        
        # Calibrer GARCH localement
        window_start = self.random_start_idx - self.calibration_window
        window_data = self.historical_data.iloc[window_start:self.random_start_idx]
        
        calibrator = GarchCalibrator()
        local_params = calibrator.fit(window_data['close'])
        
        if local_params['alpha'] + local_params['beta'] >= 0.999:
            local_params = self.garch_params
        
        initial_vol = np.sqrt(calibrator.results.conditional_volatility.iloc[-1]**2)
        initial_vol = np.clip(initial_vol, 0.0001, 0.05)
        
        self.garch_simulator = GarchSimulator(
            local_params,
            self.initial_price,
            initial_vol,
            self.volume_params
        )
        
        self.total_revenue = 0.0
        self.prices_history = [self.initial_price]
        
        warmup_needed = max(self.vol_window, self.avg_window)
        warmup_window = min(warmup_needed, self.random_start_idx - window_start)
        
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
        """Obtenir l'état actuel (Sans Sharpe Ratio)"""
        inventory_norm = self.inventory / self.initial_inventory
        time_remaining_norm = max(0.0, (self.horizon_steps - self.current_step) / self.horizon_steps)
        
        realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        rolling_sigma = self._calculate_rolling_mean(self.realized_vols_history, self.avg_window)
        rolling_sigma = max(rolling_sigma, 1e-6)
        vol_ratio = realized_vol / rolling_sigma
        
        rolling_volume = self._calculate_rolling_mean(self.volumes_history, self.avg_window)
        rolling_volume = max(rolling_volume, 1.0)
        
        liquidity_ratio = rolling_volume / self.initial_inventory
        
        current_price = self.prices_history[-1]
        price_ratio = current_price / self.initial_price
        
        # ✅ CHANGED: Removed Sharpe Ratio
        return np.array([
            inventory_norm,
            time_remaining_norm,
            vol_ratio,
            liquidity_ratio,
            price_ratio
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Exécuter une action avec récompense Implementation Shortfall"""
        
        current_price = self.prices_history[-1]
        realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        
        rolling_sigma = self._calculate_rolling_mean(self.realized_vols_history, self.avg_window)
        rolling_volume = self._calculate_rolling_mean(self.volumes_history, self.avg_window)
        
        time_remaining = self.horizon_steps - self.current_step
        
        # 1. Calculer l'exécution de l'AGENT
        # ✅ CRITICAL FIX: Action is % of REMAINING inventory
        action_pct = self.action_percentages[action] / 100.0
        
        if action_pct >= 0.99:
            # Special case: Clean sweep (avoid floating point residuals)
            agent_quantity = self.inventory
        else:
            agent_quantity = self.inventory * action_pct
        
        # Safety cap (redundant now, but good practice)
        agent_quantity = min(agent_quantity, self.inventory)
        
        temp_impact_relative = self._calculate_temporary_impact(
            agent_quantity, realized_vol, rolling_sigma, rolling_volume
        )
        execution_price = current_price * (1 - temp_impact_relative)
        agent_revenue = agent_quantity * execution_price
        
        # 2. Calculer le revenu du BASELINE (TWAP)
        twap_revenue = 0.0
        twap_quantity = 0.0
        
        if time_remaining > 0 and self.twap_inventory > 1e-6:
            twap_quantity = self.twap_inventory / time_remaining
            twap_quantity = min(twap_quantity, self.twap_inventory)
            
            twap_impact = self._calculate_temporary_impact(
                twap_quantity, realized_vol, rolling_sigma, rolling_volume
            )
            twap_execution_price = current_price * (1 - twap_impact)
            twap_revenue = twap_quantity * twap_execution_price
        
        relative_gain = agent_revenue - twap_revenue
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ NEW REWARD: Implementation Shortfall (IS) vs Arrival Price
        # ═══════════════════════════════════════════════════════════════
        
        # We want to maximize: (Execution Price - Arrival Price)
        # Reward = PnL relative to Arrival Price, normalized by Portfolio Value
        
        arrival_value = agent_quantity * self.initial_price
        is_pnl = agent_revenue - arrival_value
        
        portfolio_value = self.initial_inventory * self.initial_price
        
        # Reward in Basis Points (bps)
        reward = (is_pnl / portfolio_value) * 10000
        
        # ═══════════════════════════════════════════════════════════════
        
        self.twap_inventory -= twap_quantity
        self.total_revenue += agent_revenue
        self.inventory -= agent_quantity
        
        perm_impact_relative = self._calculate_permanent_impact(temp_impact_relative)
        perm_impact_relative = np.clip(perm_impact_relative, 0, 0.005)
        
        next_price, next_vol, next_volume = self.garch_simulator.step()
        next_price = next_price * (1 - perm_impact_relative * 0.5)
        
        if next_price > self.initial_price * 2 or next_price < self.initial_price * 0.5:
            next_price = current_price * np.random.uniform(0.99, 1.01)
        
        self.prices_history.append(next_price)
        self.volumes_history.append(next_volume)
        new_realized_vol = self._calculate_realized_volatility(np.array(self.prices_history))
        self.realized_vols_history.append(new_realized_vol)
        
        self.current_step += 1
        terminated = (self.current_step >= self.horizon_steps)
        
        # PÉNALITÉ FINALE : Inventaire NON liquidé
        if terminated and self.inventory > 1e-6:
            # Penalty: Sell remaining at 50% discount relative to current price
            penalty_quantity = self.inventory
            penalty_price = current_price * 0.0
            penalty_revenue = penalty_quantity * penalty_price
            
            penalty_arrival_value = penalty_quantity * self.initial_price
            penalty_pnl = penalty_revenue - 2*penalty_arrival_value
            
            reward += (penalty_pnl / portfolio_value) * 10000
        
        truncated = False
        next_state = self._get_state()
        
        info = {
            'quantity_sold': agent_quantity,
            'execution_price': execution_price,
            'temp_impact_relative': temp_impact_relative,
            'agent_revenue': agent_revenue,
            'twap_revenue': twap_revenue,
            'relative_gain': relative_gain,
            'inventory_remaining': self.inventory,
            'total_revenue': self.total_revenue,
            'current_price': next_price,
            'time_remaining': time_remaining,
            'reward': reward
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



