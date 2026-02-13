"""Signal smoothing filters for reducing hand tracking jitter.

Two filtering implementations:
1. OneEuroFilter
2. ExponentialMovingAverage
"""

import math
from typing import Optional


class OneEuroFilter:
    """One Euro Filter for smooth, low-latency filtering.
    
    The One Euro Filter adapts its cutoff frequency based on the speed of
    the signal, providing smooth filtering for slow movements and responsive
    tracking for fast movements.
    
    Reference: http://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """Initialize the One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag.
            beta: Cutoff slope. Higher = more responsive to speed changes.
            d_cutoff: Cutoff frequency for derivative (Hz).
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None
    
    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """Filter a new value.
        
        Args:
            x: New input value
            t: Timestamp in seconds (if None, dt=1.0 is assumed)
            
        Returns:
            Filtered value
        """
        # First call initialization
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        # Calculate time delta
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
        else:
            dt = 1.0
        
        # Avoid division by zero
        if dt <= 0:
            dt = 1.0
        
        # Calculate derivative (velocity)
        dx = (x - self.x_prev) / dt
        dx_hat = self._lowpass(dx, self.dx_prev, self._alpha(dt, self.d_cutoff))
        
        # Calculate adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the signal
        x_hat = self._lowpass(x, self.x_prev, self._alpha(dt, cutoff))
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def _alpha(self, dt: float, cutoff: float) -> float:
        """Calculate smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def _lowpass(self, x: float, x_prev: float, alpha: float) -> float:
        """Low-pass filter: exponential smoothing."""
        return alpha * x + (1.0 - alpha) * x_prev
    
    def reset(self) -> None:
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class MultiAxisFilter:
    """Apply filtering to multiple axes (e.g., x, y, z coordinates)."""
    
    def __init__(self, filter_class=OneEuroFilter, num_axes: int = 3, **filter_kwargs):
        """Initialize multi-axis filter.
        
        Args:
            filter_class: Filter class to use (OneEuroFilter or ExponentialMovingAverage)
            num_axes: Number of axes to filter
            **filter_kwargs: Arguments to pass to filter constructor
        """
        self.filters = [filter_class(**filter_kwargs) for _ in range(num_axes)]
    
    def __call__(self, values: list, t: Optional[float] = None):
        """Filter multiple values.
        
        Args:
            values: List of values to filter
            t: Optional timestamp
            
        Returns:
            List of filtered values
        """
        if len(values) != len(self.filters):
            raise ValueError(f"Expected {len(self.filters)} values, got {len(values)}")
        
        if t is not None and hasattr(self.filters[0], '__call__'):
            # OneEuroFilter accepts timestamp
            try:
                return [f(v, t) for f, v in zip(self.filters, values)]
            except TypeError:
                # Fallback if filter doesn't accept timestamp
                return [f(v) for f, v in zip(self.filters, values)]
        else:
            return [f(v) for f, v in zip(self.filters, values)]
    
    def reset(self) -> None:
        """Reset all filters."""
        for f in self.filters:
            f.reset()



"""          not used            """

class ExponentialMovingAverage:
    """Simple Exponential Moving Average filter.
    
    A lightweight alternative to OneEuroFilter for basic smoothing.
    """
    
    def __init__(self, alpha: float = 0.5):
        """Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0 to 1). Higher = more responsive, less smooth.
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.value: Optional[float] = None
    
    def __call__(self, x: float) -> float:
        """Filter a new value.
        
        Args:
            x: New input value
            
        Returns:
            Filtered value
        """
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value
    
    def reset(self) -> None:
        """Reset filter state."""
        self.value = None