from Prediction import Prediction

class MultiClassPrediction(Prediction):
    
    predictions=0;
    probabilities=0;
    
    def __init__(self,predictions,probabilities):

       Prediction.__init__(self)
       
       self.predictions=predictions
       self.probabilities=probabilities
