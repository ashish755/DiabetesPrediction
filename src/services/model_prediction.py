import pickle

import numpy as np


def get_prediction(*args):
    print("*"*100)
    gd, age, ht, hd, sh, bmi, hl, bgl = args

    test_data = np.array([gd, age, ht, hd, sh, bmi, hl, bgl]).reshape(1, -1)

    # Load the saved model using pickle
    with open('src/utils/decision_tree_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    pred = loaded_model.predict(test_data)
    print("pred:",pred)

    return pred[0]


if __name__ == "__main__":
    gd, age, ht, hd, sh, bmi, hl, bgl = (1, 70, 1, 0, 2, 35.11, 7.1, 150)
    res = get_prediction(gd, age, ht, hd, sh, bmi, hl, bgl)
    print(res)
