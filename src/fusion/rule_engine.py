"""Rule-based system for checking extreme biometric values and safety thresholds."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Alert information."""
    level: AlertLevel
    rule_name: str
    message: str
    value: float
    threshold: float


class RuleEngine:
    """
    Rule-based system for checking extreme biometric values and safety thresholds.
    
    Monitors:
    - Heart rate (HR) thresholds
    - Fall detection
    - Respiratory rate
    - Temperature
    - Movement/activity anomalies
    """
    
    def __init__(
        self,
        hr_critical_threshold: float = 180.0,
        hr_warning_threshold: float = 150.0,
        hr_low_threshold: float = 50.0,
        respiratory_rate_high: float = 40.0,
        respiratory_rate_low: float = 12.0,
        temperature_high: float = 38.5,  # Celsius
        temperature_low: float = 35.0,   # Celsius
        movement_threshold: float = 10.0,  # Accelerometer magnitude
        fall_acceleration_threshold: float = 20.0  # m/s²
    ):
        """
        Initialize rule engine with safety thresholds.
        
        Args:
            hr_critical_threshold: Critical HR threshold (default: 180 bpm)
            hr_warning_threshold: Warning HR threshold (default: 150 bpm)
            hr_low_threshold: Low HR threshold (default: 50 bpm)
            respiratory_rate_high: High respiratory rate threshold (default: 40/min)
            respiratory_rate_low: Low respiratory rate threshold (default: 12/min)
            temperature_high: High temperature threshold (default: 38.5°C)
            temperature_low: Low temperature threshold (default: 35.0°C)
            movement_threshold: Movement threshold for activity detection
            fall_acceleration_threshold: Acceleration threshold for fall detection (m/s²)
        """
        # Heart rate thresholds
        self.hr_critical_threshold = hr_critical_threshold
        self.hr_warning_threshold = hr_warning_threshold
        self.hr_low_threshold = hr_low_threshold
        
        # Respiratory rate thresholds
        self.respiratory_rate_high = respiratory_rate_high
        self.respiratory_rate_low = respiratory_rate_low
        
        # Temperature thresholds
        self.temperature_high = temperature_high
        self.temperature_low = temperature_low
        
        # Movement/activity thresholds
        self.movement_threshold = movement_threshold
        self.fall_acceleration_threshold = fall_acceleration_threshold
    
    def check_heart_rate(
        self, 
        heart_rate: float, 
        child_age_years: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Check heart rate against thresholds.
        
        Args:
            heart_rate: Heart rate in bpm
            child_age_years: Child age in years (optional, for age-adjusted thresholds)
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        # Age-adjusted thresholds if age provided
        if child_age_years is not None:
            # Infants (0-1 year): higher baseline (100-160 bpm normal)
            if child_age_years < 1:
                critical = 200.0
                warning = 170.0
                low = 80.0
            # Toddlers (1-3 years): 90-150 bpm normal
            elif child_age_years < 3:
                critical = 180.0
                warning = 160.0
                low = 70.0
            # Children (3-10 years): 70-130 bpm normal
            elif child_age_years < 10:
                critical = 160.0
                warning = 140.0
                low = 60.0
            else:
                critical = self.hr_critical_threshold
                warning = self.hr_warning_threshold
                low = self.hr_low_threshold
        else:
            critical = self.hr_critical_threshold
            warning = self.hr_warning_threshold
            low = self.hr_low_threshold
        
        if heart_rate >= critical or heart_rate <= low:
            return Alert(
                level=AlertLevel.EMERGENCY,
                rule_name="heart_rate_extreme",
                message=f"Critical heart rate: {heart_rate:.1f} bpm",
                value=heart_rate,
                threshold=critical if heart_rate >= critical else low
            )
        elif heart_rate >= warning or heart_rate <= (low + 10):
            return Alert(
                level=AlertLevel.WARNING,
                rule_name="heart_rate_elevated",
                message=f"Elevated heart rate: {heart_rate:.1f} bpm",
                value=heart_rate,
                threshold=warning if heart_rate >= warning else (low + 10)
            )
        
        return None
    
    def check_fall_detection(
        self, 
        acceleration: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Detect falls based on acceleration magnitude.
        
        Args:
            acceleration: Accelerometer data (N, 3) or (3,) - [x, y, z] in m/s²
            threshold: Optional custom threshold (default: fall_acceleration_threshold)
            
        Returns:
            Alert if fall detected, None otherwise
        """
        if threshold is None:
            threshold = self.fall_acceleration_threshold
        
        # Convert to numpy if torch tensor
        if torch.is_tensor(acceleration):
            acceleration = acceleration.cpu().numpy()
        
        # Handle different input shapes
        if acceleration.ndim == 1:
            accel_magnitude = np.linalg.norm(acceleration)
        else:
            # Compute magnitude for each sample and take max
            accel_magnitude = np.max(np.linalg.norm(acceleration, axis=-1))
        
        if accel_magnitude >= threshold:
            return Alert(
                level=AlertLevel.CRITICAL,
                rule_name="fall_detected",
                message=f"Potential fall detected: acceleration magnitude {accel_magnitude:.2f} m/s²",
                value=accel_magnitude,
                threshold=threshold
            )
        
        return None
    
    def check_respiratory_rate(
        self, 
        respiratory_rate: float
    ) -> Optional[Alert]:
        """
        Check respiratory rate against thresholds.
        
        Args:
            respiratory_rate: Respiratory rate in breaths per minute
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if respiratory_rate >= self.respiratory_rate_high:
            return Alert(
                level=AlertLevel.WARNING,
                rule_name="respiratory_rate_high",
                message=f"High respiratory rate: {respiratory_rate:.1f} breaths/min",
                value=respiratory_rate,
                threshold=self.respiratory_rate_high
            )
        elif respiratory_rate <= self.respiratory_rate_low:
            return Alert(
                level=AlertLevel.WARNING,
                rule_name="respiratory_rate_low",
                message=f"Low respiratory rate: {respiratory_rate:.1f} breaths/min",
                value=respiratory_rate,
                threshold=self.respiratory_rate_low
            )
        
        return None
    
    def check_temperature(
        self, 
        temperature: float
    ) -> Optional[Alert]:
        """
        Check body temperature against thresholds.
        
        Args:
            temperature: Body temperature in Celsius
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if temperature >= self.temperature_high:
            return Alert(
                level=AlertLevel.WARNING,
                rule_name="temperature_high",
                message=f"Elevated temperature: {temperature:.2f}°C",
                value=temperature,
                threshold=self.temperature_high
            )
        elif temperature <= self.temperature_low:
            return Alert(
                level=AlertLevel.WARNING,
                rule_name="temperature_low",
                message=f"Low temperature: {temperature:.2f}°C",
                value=temperature,
                threshold=self.temperature_low
            )
        
        return None
    
    def check_all(
        self,
        biometrics: Dict[str, float],
        acceleration: Optional[np.ndarray] = None
    ) -> List[Alert]:
        """
        Check all biometric values against thresholds.
        
        Args:
            biometrics: Dictionary with keys like 'heart_rate', 'respiratory_rate', 'temperature'
            acceleration: Optional accelerometer data for fall detection
            
        Returns:
            List of alerts (empty if no thresholds exceeded)
        """
        alerts = []
        
        # Check heart rate
        if 'heart_rate' in biometrics:
            hr_alert = self.check_heart_rate(
                biometrics['heart_rate'],
                child_age_years=biometrics.get('child_age_years')
            )
            if hr_alert:
                alerts.append(hr_alert)
        
        # Check fall detection
        if acceleration is not None:
            fall_alert = self.check_fall_detection(acceleration)
            if fall_alert:
                alerts.append(fall_alert)
        
        # Check respiratory rate
        if 'respiratory_rate' in biometrics:
            resp_alert = self.check_respiratory_rate(biometrics['respiratory_rate'])
            if resp_alert:
                alerts.append(resp_alert)
        
        # Check temperature
        if 'temperature' in biometrics:
            temp_alert = self.check_temperature(biometrics['temperature'])
            if temp_alert:
                alerts.append(temp_alert)
        
        return alerts
    
    def get_alert_level(self, alerts: List[Alert]) -> AlertLevel:
        """
        Get the highest alert level from a list of alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Highest alert level
        """
        if not alerts:
            return AlertLevel.NORMAL
        
        levels = [alert.level for alert in alerts]
        if AlertLevel.EMERGENCY in levels:
            return AlertLevel.EMERGENCY
        elif AlertLevel.CRITICAL in levels:
            return AlertLevel.CRITICAL
        elif AlertLevel.WARNING in levels:
            return AlertLevel.WARNING
        else:
            return AlertLevel.NORMAL

