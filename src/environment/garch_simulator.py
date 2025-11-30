import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
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
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        self.mean_return = returns.mean()
        centered_returns = returns - self.mean_return
        
        # ✅ TUNED: nu=10 (Tails are fat, but not "infinite variance" fat)
        self.model = arch_model(centered_returns, vol='GARCH', p=1, q=1, dist='t', rescale=False)
        self.results = self.model.fit(disp='off', show_warning=False)
        
        self.params = {
            'omega': self.results.params['omega'],
            'alpha': self.results.params['alpha[1]'],
            'beta': self.results.params['beta[1]'],
            'nu': self.results.params.get('nu', 10.0), 
            'mean_return': self.mean_return
        }
        
        self.last_price = prices.iloc[-1]
        self.last_volatility = np.sqrt(self.results.conditional_volatility.iloc[-1]**2)
        
        return self.params

    def plot_diagnostics(self, prices: pd.Series) -> None:
        """Tracer les graphiques de diagnostic"""
        if self.results is None:
            print("Vous devez d'abord calibrer le modèle avec la méthode fit()")
            return
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        returns = np.log(prices / prices.shift(1)).dropna()
        vol = self.results.conditional_volatility
        
        ax1 = axs[0]
        ax1.plot(prices.index, prices.values, 'b-', label='Prix')
        ax1.set_title('Prix et Volatilité')
        ax1b = ax1.twinx()
        ax1b.plot(vol.index, vol.values, 'r-', alpha=0.5, label='Volatilité')
        
        ax2 = axs[1]
        ax2.plot(returns.index, returns.values, 'b-', label='Rendements')
        ax2.set_title('Rendements')
        ax2b = ax2.twinx()
        ax2b.plot(vol.index, vol.values, 'r-', alpha=0.5, label='Volatilité')
        
        from scipy import stats
        ax3 = axs[2]
        residuals = self.results.resid / self.results.conditional_volatility
        stats.probplot(residuals, dist="t", sparams=(self.params['nu'],), plot=ax3)
        ax3.set_title(f'QQ-Plot des résidus (Student-t, nu={self.params["nu"]:.2f})')
        
        plt.tight_layout()
        plt.show()


class VolumeModelCalibrator:
    """
    Classe pour calibrer un modèle volume basé sur l'échantillonnage empirique conditionnel
    """
    def __init__(self):
        self.volume_bins = {}
        self.vol_quantiles = None
        
    def calibrate(self, volumes: np.ndarray, conditional_variances: np.ndarray) -> Dict:
        """
        Calibrer le modèle volume par binning de volatilité
        """
        vols = np.sqrt(conditional_variances)
        self.vol_quantiles = np.quantile(vols, [0.2, 0.4, 0.6, 0.8])
        bin_indices = np.digitize(vols, self.vol_quantiles)
        
        self.volume_bins = {}
        for i in range(5):
            bin_volumes = volumes[bin_indices == i]
            self.volume_bins[i] = bin_volumes
            
        return {
            'vol_quantiles': self.vol_quantiles,
            'volume_bins': self.volume_bins
        }


class GarchSimulator:
    """
    Simulateur de prix et volumes amélioré (Student-t + Jumps + Empirical Volume)
    """
    def __init__(self, 
                 garch_params: dict, 
                 initial_price: float, 
                 initial_volatility: Optional[float] = None,
                 volume_params: Optional[Dict] = None):
        
        self.omega = garch_params['omega']
        self.alpha = garch_params['alpha']
        self.beta = garch_params['beta']
        self.nu = garch_params.get('nu', 10.0)
        
        # ✅ CRITICAL FIX: Force Zero Drift for Execution Simulation
        # We ignore the historical mean return because intraday price is a Martingale.
        # This prevents the "30% drift" artifact.
        self.mean_return = 0.0 
        
        self.price = initial_price
        self.volume_params = volume_params
        
        uncond_variance = self.omega / (1 - self.alpha - self.beta)
        self.last_variance = initial_volatility**2 if initial_volatility else uncond_variance
        self.last_return = 0.0
        
        # ✅ TUNED: Reduced Jump Probability
        self.jump_prob = 0.0005  # 0.05% chance (approx once per 33 hours)
        self.jump_size_mean = 0.0
        self.jump_size_std = 0.015 # 1.5% jump size

    def _generate_next_return(self) -> float:
        """
        Générer le prochain rendement (GARCH Student-t + Jumps)
        """
        current_variance = (self.omega + 
                           self.alpha * self.last_return**2 + 
                           self.beta * self.last_variance)
        
        # Student-t innovations
        std_t_scale = np.sqrt((self.nu - 2) / self.nu)
        z = np.random.standard_t(self.nu) * std_t_scale
        
        current_volatility = np.sqrt(current_variance)
        
        # ✅ SAFETY: Cap volatility to realistic max (2% per minute is already a crash)
        current_volatility = min(current_volatility, 0.02)
        
        # ✅ CRITICAL FIX: Martingale Correction (Jensen's Inequality)
        # We subtract 0.5 * sigma^2 to ensure E[Price_t] = Price_{t-1}
        # Without this, the price naturally drifts UP due to volatility.
        drift_correction = -0.5 * (current_volatility**2) # Use capped vol for correction
        
        current_return = self.mean_return + drift_correction + current_volatility * z
        
        # Jump Component
        if np.random.random() < self.jump_prob:
            jump = np.random.normal(self.jump_size_mean, self.jump_size_std)
            current_return += jump
        
        # ✅ SAFETY: Circuit Breaker
        # Cap single-step return to +/- 5% to prevent numerical explosion
        # (5% in 1 minute is still a massive crash, but keeps math stable)
        current_return = np.clip(current_return, -0.05, 0.05)
        
        self.last_variance = current_variance
        self.last_return = current_return # mean_return is 0, so this is correct
        
        return current_return
    
    def _generate_volume(self) -> float:
        """
        Générer un volume par échantillonnage empirique conditionnel
        """
        if self.volume_params is None:
            return 1.0
        
        current_vol = np.sqrt(self.last_variance)
        vol_quantiles = self.volume_params['vol_quantiles']
        bin_idx = np.digitize(current_vol, vol_quantiles)
        
        possible_volumes = self.volume_params['volume_bins'][bin_idx]
        volume = np.random.choice(possible_volumes)
        
        volume *= np.random.uniform(0.95, 1.05)
        volume *= 0.8 # Conservative scaling
        
        return max(0.01, volume)

    def step(self) -> Tuple[float, float, float]:
        """Simuler un pas de temps"""
        current_return = self._generate_next_return()
        self.price = self.price * np.exp(current_return)
        current_volatility = np.sqrt(self.last_variance)
        current_volume = self._generate_volume()
        
        return self.price, current_volatility, current_volume
    
    def simulate_path(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simuler une trajectoire complète"""
        prices = np.zeros(n_steps + 1)
        volatilities = np.zeros(n_steps + 1)
        volumes = np.zeros(n_steps) if self.volume_params else None
        
        prices[0] = self.price
        volatilities[0] = np.sqrt(self.last_variance)
        
        for i in range(n_steps):
            prices[i+1], volatilities[i+1], volume = self.step()
            if volumes is not None:
                volumes[i] = volume
            
        return prices, volatilities, volumes
    
    def reset(self, price: Optional[float] = None, volatility: Optional[float] = None) -> None:
        if price is not None: self.price = price
        if volatility is not None: self.last_variance = volatility**2
        self.last_return = 0.0


def calibrate_full_model(data: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Calibrer GARCH et Volume"""
    garch_calibrator = GarchCalibrator()
    garch_params = garch_calibrator.fit(data['close'])
    
    conditional_variances = garch_calibrator.results.conditional_volatility ** 2
    volumes = data['volume'].values[len(data) - len(conditional_variances):]
    
    volume_calibrator = VolumeModelCalibrator()
    volume_params = volume_calibrator.calibrate(volumes, conditional_variances.values)
    
    return garch_params, volume_params


if __name__ == "__main__":
    # Test rapide
    import pandas as pd
    try:
        data = pd.read_csv('../../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv', 
                        index_col=0, parse_dates=True)
        garch_params, volume_params = calibrate_full_model(data)
        
        initial_price = data['close'].iloc[-1]
        initial_vol = np.sqrt(garch_params['omega'] / (1 - garch_params['alpha'] - garch_params['beta']))
        
        sim = GarchSimulator(garch_params, initial_price, initial_vol, volume_params)
        p, v, vol = sim.simulate_path(1000)
        
        print(f"Simulation OK. Max Volume: {np.max(vol):.2f}")
    except Exception as e:
        print(f"Erreur test: {e}")