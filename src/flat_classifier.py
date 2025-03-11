from sklearn.base import BaseEstimator

class FlatClassifier(BaseEstimator):

    def __init__(self, local_classifier):
        self.local_classifier = local_classifier 

    def fit(self, X, y):
        y = ["::HiClass::Separator::".join(i) for i in y]
        self.local_classifier.fit(X, y)
        return self
    
    def predict(self, X):
        return [i.split('::HiClass::Separator::') for i in self.local_classifier.predict(X)]