import logging
import matplotlib.pyplot as plt

def __logging__(model,X_test,y_test):
        logging.basicConfig(filename='predictions.log',level=logging.INFO)
        for i in range (X_test.shape[0]):
            instance=X_test.iloc[i,:].values.reshape(1,-1)
            prediction=model.predict(instance)
            logging.info(f'Inst.{i}-PredClass:{prediction[0]}, RealClass:{y_test.iloc[i]}')


