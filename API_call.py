import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import joblib

# -------------------------
# Base Class
# -------------------------
class FluidSimulator:
    def __init__(self, model_path, fs=10, duration=600):
        self.fs = fs
        self.duration = duration
        self.time = np.arange(duration) / fs
        self.model = joblib.load(model_path)
        self.requiredVars = self.model.feature_names_in_

    def simulate_base_signals(self):
        """Default (normal) flow and pressure without faults."""
        flow = 1.2 + 0.02*np.random.randn(self.duration)
        pressure = 1.18 + 0.02*np.random.randn(self.duration)
        return flow, pressure

    def inject_faults(self, flow, pressure):
        """To be overridden by subclasses with fluid-specific fault logic."""
        return flow, pressure

    def extract_features(self, flow, pressure):
        """Generic feature extraction (works for any fluid)."""
        feats = {}
        feats["flowMean"] = np.mean(flow)
        feats["flowStd"] = np.std(flow)
        feats["flowRMS"] = np.sqrt(np.mean(flow**2))
        feats["flowPeak2Peak"] = np.ptp(flow)
        feats["flowSkew"] = skew(flow)
        feats["flowKurt"] = kurtosis(flow)

        # FFT dominant frequency
        L = len(flow)
        Y = np.abs(fft(flow - np.mean(flow)))
        P1 = Y[:L//2]
        idx = np.argmax(P1[1:]) + 1
        feats["flowDominantFreq"] = idx * self.fs / L

        # Pressure stats
        feats["pressureMean"] = np.mean(pressure)
        feats["pressureStd"] = np.std(pressure)
        feats["pressureRMS"] = np.sqrt(np.mean(pressure**2))
        feats["pressurePeak2Peak"] = np.ptp(pressure)
        feats["pressureSkew"] = skew(pressure)
        feats["pressureKurt"] = kurtosis(pressure)

        # Combined
        feats["flowPressureCorr"] = np.corrcoef(flow, pressure)[0,1]
        feats["absDiffMean"] = np.mean(np.abs(flow-pressure))
        feats["rmsRatio"] = feats["flowRMS"]/feats["pressureRMS"]

        # Select only required
        feat_array = [feats[var] for var in self.requiredVars]
        return pd.DataFrame([feat_array], columns=self.requiredVars)

    def predict_fault(self, features):
        """Run model prediction."""
        pred = self.model.predict(features)[0]
        conf = np.max(self.model.predict_proba(features)[0])
        return pred, conf

    def run(self):
        """Complete pipeline: simulate → inject faults → extract → predict."""
        flow, pressure = self.simulate_base_signals()
        flow, pressure = self.inject_faults(flow, pressure)
        features = self.extract_features(flow, pressure)
        pred, conf = self.predict_fault(features)
        return {"prediction": pred, "confidence": conf}




class WaterSimulator(FluidSimulator):
    def inject_faults(self, flow, pressure):
        # Example fault: leakage → lower flow, oscillating pressure
        fault_range = range(100, 200)
        idx = np.array(list(fault_range))
        t = self.time[idx] - self.time[idx[0]]
        flow[idx] += 0.5*np.sin(2*np.pi*0.3*t) - 0.4
        pressure[idx] += 0.6*np.sin(2*np.pi*1.2*t)
        return flow, pressure

class OilSimulator(FluidSimulator):
    def inject_faults(self, flow, pressure):
        # Example fault: friction → noisy flow/pressure
        fault_range = range(250, 300)
        idx = np.array(list(fault_range))
        flow[idx] += 0.3*np.random.randn(len(idx))
        pressure[idx] += 1.0*np.random.randn(len(idx))
        return flow, pressure

sim = WaterSimulator(model_path="", fs=10, duration=600)
results = sim.run()
print(results)
