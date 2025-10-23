 
"""
Rate Limiter Module

Single Responsibility: Prevent API rate limit violations
- Track API call frequency
- Enforce rate limits (100 calls per 60 seconds)
- Provide decorator for automatic rate limiting

Spotify API limits: ~180 requests per minute (we use conservative 100)
"""

import time
from threading import Lock
from collections import deque
from functools import wraps
from typing import Callable, Any


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Tracks API calls and enforces rate limits to prevent 429 errors.
    Thread-safe implementation using locks.
    """
    
    def __init__(self, max_calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = Lock()
    
    def _clean_old_calls(self) -> None:
        """Remove calls outside the current time window."""
        now = time.time()
        cutoff = now - self.period
        
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()
    
    def wait_if_needed(self) -> None:
        """
        Wait if rate limit would be exceeded.
        
        Blocks until a call can be made without exceeding the limit.
        """
        with self.lock:
            self._clean_old_calls()
            
            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = self.calls[0]
                wait_time = oldest_call + self.period - time.time()
                
                if wait_time > 0:
                    print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    self._clean_old_calls()
            
            # Record this call
            self.calls.append(time.time())
    
    def get_current_usage(self) -> dict:
        """
        Get current rate limit usage.
        
        Returns:
            Dict with calls_made, calls_remaining, window_reset_in
        """
        with self.lock:
            self._clean_old_calls()
            calls_made = len(self.calls)
            calls_remaining = self.max_calls - calls_made
            
            if self.calls:
                oldest_call = self.calls[0]
                window_reset_in = oldest_call + self.period - time.time()
            else:
                window_reset_in = self.period
            
            return {
                'calls_made': calls_made,
                'calls_remaining': calls_remaining,
                'window_reset_in': max(0, window_reset_in),
                'max_calls': self.max_calls,
                'period': self.period
            }


def rate_limited(limiter: RateLimiter) -> Callable:
    """
    Decorator to rate limit function calls.
    
    Args:
        limiter: RateLimiter instance to use
        
    Returns:
        Decorated function that respects rate limits
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator