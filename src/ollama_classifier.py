import json, random
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class OllamaClassifier(BaseEstimator):

    def __init__(self, model):
        self.model = ChatOllama(model=model, format='json')

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.labels_ = [item.split('::HiClass::Separator::')[-1] for item in self.classes_.tolist()]
        self.template = ChatPromptTemplate.from_template("""
                Based on the article content:\n\n
                {text}\n\n
                Classify the content into one correct research area:
                {labels}
                Return a JSON object like ['Research Area': '']."""
            ) | self.model
        return self

    def predict(self, X):
        predictions = []
        for text in X:

            pred = ''
            self.classify_content(self.labels_)
            content = self.template.invoke({'text': text, 'labels': '; '.join(self.labels_)})

            try:
                content = content.dict()
                pred = json.loads(content['content'])['Research Area']
                idx = self.labels_.index(pred)
            except:  
                pred = random.choice(self.labels_)
                idx = self.labels_.index(pred)

            predictions.append(self.classes_[idx])

        return np.array(predictions)

        
    def classify_content(self, labels):

        self.model = self.model.bind(
            tools = [{
                'name': 'classify_content',
                'description': 'Predict the research area for a given article content',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'prediction': {
                            'type': 'array',
                            'description': 'The predicted reserach areas.',
                            'items': {
                                'type': 'string',
                                'enum': labels
                            },
                        }
                    },
                    'required': ['prediction']
                }
            }], 
            function_call={'name': 'classify_content'}
        )