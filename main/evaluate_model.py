import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

import recognition_backend
from deepface.commons import distance as dst

# Find the accuracy given the predicted and target values
def find_accuracy(pred, true):
    accuracy = sum([1 if pred_face == true_face else 0 for pred_face, true_face in zip(pred, true)]) / len(pred)
    return accuracy

# Evaluate a face recognition model using k-fold cross validation
def evaluate_model(
          db_path,
          splits=5, 
          model_name="Facenet512", 
          distance_metric="cosine", 
          enforce_detection=True, 
          detector_backend="skip", 
          align=False, 
          normalization="base", 
          silent=False
    ):
    kwargs = {key: value for key, value in locals().items() if key not in ["distance_metric", "splits"]}

    # Get the representations of the faces
    representations = recognition_backend.build_representations(**kwargs)
    df = pd.DataFrame(representations, columns=["closest_match", f"{model_name}_representation"])
    print(df)

    # Get the names of the faces, together with the vectors representing them
    labels = df["closest_match"].values
    X = np.array(df[f"{model_name}_representation"].values)
    y = np.array([identity.split("\\")[1] for identity in labels])

    kfold = StratifiedKFold(n_splits=splits)
    cnn_predictions, cnn_scores = {"pred":[], "true":[]}, []

    # Split the dataset into k folds
    for train, test in kfold.split(X, y):
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]

            y_pred = []

            # Predict the name of each face based on its vector representation using the given metric
            for source_instance in X_test:
                if distance_metric == "cosine":
                    pred_index = np.argmin([dst.findCosineDistance(source_instance, target_instance) for target_instance in X_train])
                elif distance_metric == "euclidean":
                    pred_index = np.argmin([dst.findEuclideanDistance(source_instance, target_instance) for target_instance in X_train])
                elif distance_metric == "euclidean_l2":
                    pred_index = np.argmin([dst.findEuclideanDistance(dst.l2_normalize(source_instance), dst.l2_normalize(target_instance)) for target_instance in X_train])
                else:
                    raise ValueError(f"invalid distance metric passes - {distance_metric}")
                prediction = y_train[pred_index]
                y_pred.append(prediction)

            # Save these scores
            cnn_scores.append(find_accuracy(y_pred, y_test))
            cnn_predictions["pred"].extend(y_pred)
            cnn_predictions["true"].extend(y_test)
        
    # Print the average 5-fold validation scores
    accuracy = round(sum(cnn_scores)/len(cnn_scores), 5) * 100
    print(f"Average score for {model_name}: ", accuracy, "%")

    # Show the confusion matrix
    ax = plt.axes()
    plt.suptitle(f"Confusion matrix for {model_name}")
    plt.title(f"Accuracy: {accuracy}%", fontsize=10)
    ConfusionMatrixDisplay.from_predictions(
         cnn_predictions["true"], 
         cnn_predictions["pred"], 
         xticks_rotation="vertical", 
         include_values=False, 
         normalize="true", 
         display_labels=["" for i in range(len(set(cnn_predictions["true"])))], 
         ax=ax)
    ax.tick_params(axis='both', which='both', length=0)
    plt.show()


evaluate_model(
          db_path="./datasets/lfw50",
          splits=5, 
          model_name="Facenet512", 
          distance_metric="cosine", 
          enforce_detection=True, 
          detector_backend="skip", 
          align=False, 
          normalization="base", 
          silent=False
)