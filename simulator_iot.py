"""
TunnelSpec AI — IoT Sensor Simulator
Simulates MQTT-style telemetry from a live TBM tunnel drive.
Sensors: TBM Face Pressure, Cutterhead Torque, Advance Rate,
         Vibration, Water Ingress, Tail Grout Pressure, Surface Settlement,
         Annulus Grout Volume.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class SensorConfig:
    name: str
    unit: str
    nominal: float
    std_dev: float
    alert_high: float
    alert_low: float
    trend_rate: float   # linear drift per tick
    line_color: str
    fill_color: str


SENSOR_CONFIGS: Dict[str, SensorConfig] = {
    "tbm_face_pressure": SensorConfig(
        name="TBM Face Pressure",       unit="bar",
        nominal=2.5,    std_dev=0.12,   alert_high=3.8,  alert_low=0.8,
        trend_rate=0.002,
        line_color="#2196F3", fill_color="rgba(33,150,243,0.12)",
    ),
    "cutterhead_torque": SensorConfig(
        name="Cutterhead Torque",       unit="kN·m",
        nominal=450.0,  std_dev=25.0,   alert_high=620.0, alert_low=80.0,
        trend_rate=0.5,
        line_color="#FF9800", fill_color="rgba(255,152,0,0.12)",
    ),
    "advance_rate": SensorConfig(
        name="Advance Rate",            unit="mm/min",
        nominal=38.0,   std_dev=4.0,    alert_high=75.0,  alert_low=5.0,
        trend_rate=-0.04,
        line_color="#4CAF50", fill_color="rgba(76,175,80,0.12)",
    ),
    "vibration_rms": SensorConfig(
        name="Vibration (RMS)",         unit="mm/s",
        nominal=1.2,    std_dev=0.25,   alert_high=3.0,   alert_low=0.0,
        trend_rate=0.003,
        line_color="#F44336", fill_color="rgba(244,67,54,0.12)",
    ),
    "water_ingress": SensorConfig(
        name="Water Ingress",           unit="L/min",
        nominal=0.6,    std_dev=0.15,   alert_high=2.5,   alert_low=0.0,
        trend_rate=0.001,
        line_color="#00BCD4", fill_color="rgba(0,188,212,0.12)",
    ),
    "grout_pressure": SensorConfig(
        name="Tail Grout Pressure",     unit="bar",
        nominal=1.8,    std_dev=0.08,   alert_high=3.0,   alert_low=0.5,
        trend_rate=0.001,
        line_color="#9C27B0", fill_color="rgba(156,39,176,0.12)",
    ),
    "surface_settlement": SensorConfig(
        name="Surface Settlement",      unit="mm",
        nominal=2.1,    std_dev=0.20,   alert_high=8.0,   alert_low=0.0,
        trend_rate=0.005,
        line_color="#FF5722", fill_color="rgba(255,87,34,0.12)",
    ),
    "annulus_grout_volume": SensorConfig(
        name="Annulus Grout Volume",    unit="L/ring",
        nominal=185.0,  std_dev=12.0,   alert_high=350.0, alert_low=80.0,
        trend_rate=0.0,
        line_color="#607D8B", fill_color="rgba(96,125,139,0.12)",
    ),
}


class TBMSimulator:
    """
    Simulates a TBM (Tunnel Boring Machine) sensor data stream.
    Generates readings with realistic noise, geological trend drift,
    and occasional operational spikes to mimic MQTT field telemetry.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.tick: int = 0
        self.drive_chainage: float = 0.0  # metres advanced
        self.ring_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_value(self, cfg: SensorConfig, tick: int) -> float:
        """Produce a single sensor reading with noise + trend + spike."""
        base = cfg.nominal + cfg.trend_rate * tick
        noise = np.random.normal(0, cfg.std_dev)

        # Low-probability operational spike (e.g. probe impact, valve event)
        spike = 0.0
        if np.random.random() < 0.02:
            spike = cfg.std_dev * np.random.uniform(3, 6) * np.random.choice([-1, 1])

        # Sinusoidal geology variation (stratified ground)
        geo = cfg.std_dev * 0.5 * np.sin(tick * 0.08)

        value = base + noise + spike + geo
        return round(float(np.clip(value, max(cfg.alert_low * 0.5, 0), cfg.alert_high * 1.25)), 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reading(self) -> Dict[str, Any]:
        """Return a single timestamped reading from all sensors."""
        self.tick += 1
        self.drive_chainage += float(np.random.uniform(0.03, 0.08))
        if self.tick % 14 == 0:
            self.ring_count += 1

        reading: Dict[str, Any] = {
            "timestamp":    datetime.now(),
            "chainage_m":   round(self.drive_chainage, 2),
            "ring_no":      self.ring_count,
        }
        for key, cfg in SENSOR_CONFIGS.items():
            reading[key] = self._generate_value(cfg, self.tick)
        return reading

    def get_historical_data(self, n_points: int = 120) -> pd.DataFrame:
        """Generate a seeded historical dataset for initial dashboard render."""
        np.random.seed(99)
        records = []
        base_time = datetime.now() - timedelta(minutes=n_points)
        chainage, ring = 0.0, 0

        for i in range(n_points):
            chainage += float(np.random.uniform(0.03, 0.08))
            if i % 14 == 0:
                ring += 1
            row: Dict[str, Any] = {
                "timestamp":  base_time + timedelta(minutes=i),
                "chainage_m": round(chainage, 2),
                "ring_no":    ring,
            }
            for key, cfg in SENSOR_CONFIGS.items():
                row[key] = self._generate_value(cfg, i)
            records.append(row)

        np.random.seed(42)  # restore main seed
        return pd.DataFrame(records)

    def get_alerts(self, reading: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check a reading against alert thresholds. Returns list of active alerts."""
        alerts = []
        for key, cfg in SENSOR_CONFIGS.items():
            value = float(reading.get(key, 0))
            if value >= cfg.alert_high:
                severity = "CRITICAL" if value >= cfg.alert_high * 1.1 else "WARNING"
                alerts.append({
                    "sensor":    cfg.name,
                    "value":     value,
                    "unit":      cfg.unit,
                    "threshold": cfg.alert_high,
                    "severity":  severity,
                    "message":   (
                        f"{cfg.name} {severity}: {value} {cfg.unit} "
                        f"(limit: {cfg.alert_high} {cfg.unit})"
                    ),
                })
        return alerts

    def get_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Derive headline drive-performance KPIs from historical data."""
        if df.empty:
            return {}
        return {
            "total_drive_m":      round(float(df["chainage_m"].max()), 1),
            "rings_built":        int(df["ring_no"].max()),
            "avg_advance_rate":   round(float(df["advance_rate"].mean()), 1),
            "avg_face_pressure":  round(float(df["tbm_face_pressure"].mean()), 3),
            "max_settlement":     round(float(df["surface_settlement"].max()), 2),
            "uptime_pct":         97.3,   # representative simulated value
        }
