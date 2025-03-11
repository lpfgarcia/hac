from sklearn.svm import SVC

class SVM(SVC):
    def __init__(self, C=1.0, kernel='rbf', probability=True):
        super().__init__(C=C, kernel=kernel, probability=probability)