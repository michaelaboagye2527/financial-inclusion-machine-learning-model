
class ThresholdXGBClassifier:

    def __init__(self, trained_pipeline, threshold=0.40):
        self.pipeline = trained_pipeline
        self.threshold = threshold

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        y_probs = self.pipeline.predict_proba(X)[:, 1]
        return (y_probs >= self.threshold).astype(int)
