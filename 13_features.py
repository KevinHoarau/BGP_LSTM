from utils import *
import numpy as np
from sklearn.model_selection import KFold
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data = {
    'NA': loadData("data/no_anomaly/"),
    'A': loadData("data/anomaly/")
}

device = 'cuda'

seeds = range(33)

cv = KFold(3, shuffle=True, random_state=1)

events = np.array(list(data["A"].keys()))

accAll = []
f1All = []
prAll = []
recAll = []

k=0
for a,b in cv.split(events):
    
    k += 1
    
    accs = []
    f1s = []
    prs = []
    recs = []
    
    for s in seeds:
        
        clf = NeuralNetClassifier(
            RNNClassifier,
            verbose=0,
            iterator_train__shuffle=False,
            batch_size=1000,
            optimizer=torch.optim.Adam,
            train_split=None,
            module__input_dim = 13,
            device = device,
            lr = 0.1,
            max_epochs = 50,
            module__dropout = 0.25,
            module__num_layers = 1,
            module__num_units = 32,
            module__seed = s,
            optimizer__weight_decay = 0,
        )

        e_train = list(events[a])
        e_test = list(events[b])

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for e in events:

            X_0 = data["A"][e][:60]
            X_1 = data["NA"][e][:60]

            if e in e_test:
                X_test.append(X_0)
                X_test.append(X_1)
                Y_test += [0,1]
            if e in e_train:
                X_train.append(X_0)
                X_train.append(X_1)
                Y_train += [0,1]

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Train

        clf.fit(X_train, Y_train)

        # Test

        pred_test = clf.predict(X_test)
        accs.append(accuracy_score(Y_test, pred_test))
        f1s.append(f1_score(Y_test, pred_test, zero_division=0))
        prs.append(precision_score(Y_test, pred_test, zero_division=0))
        recs.append(recall_score(Y_test, pred_test, zero_division=0))
    
    print("########")
    print("Fold",k)
    print("Accuracy: mean:%.2f, std:%.2f"%(np.mean(accs), np.std(accs)))
    print("F1 score: mean:%.2f, std:%.2f"%(np.mean(f1s), np.std(f1s)))
    print("Precision: mean:%.2f, std:%.2f"%(np.mean(prs), np.std(prs)))
    print("Recall: mean:%.2f, std:%.2f"%(np.mean(recs), np.std(recs)))
    
    accAll += accs
    f1All += f1s
    prAll += prs
    recAll += recs

print("########")
print("All")
print("Accuracy: mean:%.2f, std:%.2f"%(np.mean(accAll), np.std(accAll)))
print("F1 score: mean:%.2f, std:%.2f"%(np.mean(f1All), np.std(f1All)))
print("Precision: mean:%.2f, std:%.2f"%(np.mean(prAll), np.std(prAll)))
print("Recall: mean:%.2f, std:%.2f"%(np.mean(recAll), np.std(recAll)))