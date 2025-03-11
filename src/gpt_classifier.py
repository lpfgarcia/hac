import random, openai, json
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels

class GPTClassifier(BaseEstimator):

    def __init__(self, model, key):
        self.model = model
        self.key = key
        openai.api_key = self.key

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.labels_ = [item.split('::HiClass::Separator::')[-1] for item in self.classes_.tolist()]
        return self

    def predict(self, X):
        predictions = []
        for text in X:

            pred = ''
            text = (f'Classify the article content into one correct research area:\n {text}')
            completion = openai.chat.completions.create(
                model = self.model,
                messages = [{'role': 'user', 'content': text}],
                tools = self.classify_content(self.labels_),
                tool_choice = {'type': 'function', 'function': {'name': 'classify_content'}}
            )

            try:
                content = completion.choices[0].message.tool_calls[0].function.arguments
                pred = json.loads(content)['prediction'][0]
                idx = self.labels_.index(pred)
            except:  
                pred = random.choice(self.labels_)
                idx = self.labels_.index(pred)

            predictions.append(self.classes_[idx])

        return np.array(predictions)

    def classify_content(self, labels):

        return [{
                'type': 'function',
                'function': {
                    'name': 'classify_content',
                    'description': 'Predict the research area for a given article content',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'prediction': {
                                'type': 'array',
                                'items': {
                                    'type': 'string',
                                    'enum': labels
                                },
                                'description': 'The predicted reserach areas.'
                            }
                        },
                        'required': [
                            'prediction'
                        ]
                    }
                }
        }]