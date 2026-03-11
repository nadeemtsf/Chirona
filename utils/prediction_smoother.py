"""
Temporal smoothing for machine learning predictions.

Applies a moving window over recent predictions to filter out single-frame
flickers and report a stable, dominant gesture prediction.
"""
from collections import Counter, deque
from core.config_manager import config_mgr

class PredictionSmoother:
    def __init__(self):
        # Initial deque will resize dynamically in add_prediction if window changes
        self.prediction_history = deque(maxlen=config_mgr.get('SMOOTHING_WINDOW_SIZE'))

    def add_prediction(self, new_prediction):
        # Resize deque safely if config window size was altered
        current_max = self.prediction_history.maxlen
        target_max = config_mgr.get('SMOOTHING_WINDOW_SIZE')
        
        if current_max != target_max:
            new_deque = deque(maxlen=target_max)
            for item in self.prediction_history:
                new_deque.append(item)
            self.prediction_history = new_deque
            
        self.prediction_history.append(new_prediction)
    
    def get_stable_prediction(self):
        if not self.prediction_history:
            return None
        prediction, count = Counter(self.prediction_history).most_common(1)[0]
        ratio = count / len(self.prediction_history)
        if ratio >= config_mgr.get('SMOOTHING_DOMINANCE_THRESHOLD'):
            return prediction
        return None

    def clear(self):
        self.prediction_history.clear()