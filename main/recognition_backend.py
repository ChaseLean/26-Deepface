# common dependencies
import os
import time
import pickle
import warnings
import logging

# 3rd party dependencies
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf

# package dependencies
from deepface.DeepFace import represent
from deepface.commons import functions, distance as dst
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    SFace,
)
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, distance as dst

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

def build_model(model_name):

    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes
    Returns:
            built deepface model
    """

    # singleton design pattern
    global model_obj

    models = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "DeepID": DeepID.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFace.load_model,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]

def build_extended_models(actions):

    # validate actions
    if isinstance(actions, str):
        actions = (actions,)

    actions = list(actions)

    # build models
    extended_models = {}
    if "emotion" in actions:
        extended_models["emotion"] = build_model("Emotion")

    if "age" in actions:
        extended_models["age"] = build_model("Age")

    if "gender" in actions:
        extended_models["gender"] = build_model("Gender")

    if "race" in actions:
        extended_models["race"] = build_model("Race")
    
    return extended_models

def build_representations(
    db_path,
    model_name,
    enforce_detection,
    detector_backend,
    align,
    normalization,
    silent,
):
    
    target_size = functions.find_target_size(model_name=model_name)

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()

    if os.path.exists(db_path + "/" + file_name):

        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print("There are ", len(representations), " representations found in ", file_name)

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                    or (".JPG" in file.lower())
                    or (".JPEG" in file.lower())
                    or (".PNG" in file.lower())
                ):
                    exact_path = r + "\\" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
        )
        for index in pbar:
            employee = employees[index]

            kwargs = {
                "target_size": target_size,
                "detector_backend": detector_backend,
                "grayscale": False,
                "enforce_detection": enforce_detection,
                "align": align 
            }
            
            try:
                img_objs = functions.extract_faces(employee, **kwargs)
            except ValueError:
                print(f"No face detected in {employee}. Ignoring this image.")
                img_objs = []
            except cv2.error:
                print(f"Image alignment failed on {employee}. The face is likely too close to the border of the image. Retrying with border padding.")
                try:
                    image = cv2.imread(employee)
                    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    img_objs = functions.extract_faces(image, **kwargs)
                    print("Successfully extracted image.")
                except cv2.error:
                    print(f"Solve failed. Ignoring {employee} from database. To extract this face, try using another detector.")
            
            # choosing only the largest face to be saved (cannot have two people with the same name)
            if len(img_objs) > 0:
                max_index = np.argmax([obj[1]["w"] for obj in img_objs])
                img_objs = np.array([img_objs[max_index]])

            for img_content, _, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new image_paths in your database."
            )

    return representations


def analyse_face(extended_models, img_content, img_region, actions, silent=True):
        
        obj = {}

        # resizing image to 224x224 for facial attribute models
        img_content = np.array([cv2.resize(img_content[0], (224, 224))])


        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            # facial attribute analysis
            pbar = tqdm(range(0, len(actions)), desc="Finding actions", disable=silent)
            for index in pbar:
                action = actions[index]
                pbar.set_description(f"Action: {action}")

                if action == "emotion":
                    img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)

                    emotion_predictions = extended_models["emotion"].predict(img_gray, verbose=0)[0, :]

                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {}

                    for i, emotion_label in enumerate(Emotion.labels):
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        obj["emotion"][emotion_label] = emotion_prediction

                    obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

                elif action == "age":
                    age_predictions = extended_models["age"].predict(img_content, verbose=0)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)
                    # int cast is for exception - object of type 'float32' is not JSON serializable
                    obj["age"] = int(apparent_age)

                elif action == "gender":
                    gender_predictions = extended_models["gender"].predict(img_content, verbose=0)[0, :]
                    obj["gender"] = {}
                    for i, gender_label in enumerate(Gender.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj["gender"][gender_label] = gender_prediction

                    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

                elif action == "race":
                    race_predictions = extended_models["race"].predict(img_content, verbose=0)[0, :]
                    sum_of_predictions = race_predictions.sum()

                    obj["race"] = {}
                    for i, race_label in enumerate(Race.labels):
                        race_prediction = 100 * race_predictions[i] / sum_of_predictions
                        obj["race"][race_label] = race_prediction

                    obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]

                # -----------------------------
                # mention facial areas
                obj["region"] = img_region

        return obj

def init(
        db_path,
        model_name,
        enforce_detection,
        detector_backend,
        actions,
        align,
        normalization,
        silent
):
    representations = build_representations(
        db_path=db_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        normalization=normalization,
        silent=silent
    )
    if not silent and len(actions) > 0:
        print("Building models...")
    extended_models = build_extended_models(actions=actions)
    return representations, extended_models

def find(
    img_path,
    representations,
    extended_models=[],
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):

    """
    This function applies verification several times and find the image_paths in a database

    Parameters:
            img_path: exact image path, numpy array (BGR) or based64 encoded image.
            Source image can have many faces. Then, result will be the size of number of
            faces in the source image.

            db_path (string): You should store some image files in a folder and pass the
            exact folder path to this. A database image can also have many faces.
            Then, all detected faces in db side will be considered in the decision.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
            Dlib, ArcFace, SFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (bool): The function throws exception if a face could not be detected.
            Set this to True if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

            silent (boolean): disable some logging and progress bars

    Returns:
            This function returns list of pandas data frame. Each item of the list corresponding to
            an identity in the img_path.
    """

        # -------------------------------
    
    tic = time.time()

    # ---------------------------------

    # now, we got representations for facial database
    df = pd.DataFrame(representations, columns=["closest_match", f"{model_name}_representation"])

    if not silent:
        print("Extracting faces...")

    target_size = functions.find_target_size(model_name=model_name)

    # img path might have more than once face
    try:
        target_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    except ValueError:
        if not silent:
            print("Face could not be detected. Please make sure that the picture contains a face.")
        target_objs = []
    except cv2.error:
        if not silent:
            print("Image alignment failed. The face is likely too close to the border of the image.")
        target_objs = []

    resp_obj = []

    for target_img, target_region, _ in tqdm(target_objs, desc="Analysing faces", disable=(silent or len(target_objs) == 0)):
        target_embedding_obj = represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        actions = list(extended_models.keys())

        if len(actions) > 0:
            analysis = analyse_face(extended_models=extended_models, img_content=target_img, img_region=target_region, actions=actions)
            key_dict = {"gender": "dominant_gender",
                        "emotion": "dominant_emotion",
                        "age": "age",
                        "race": "dominant_race"}
            for action in actions:
                result_df[action] = analysis[key_dict[action]]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df['name'] = [identity.split("\\")[1] for identity in result_df["closest_match"].values]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("Find function lasts ", toc - tic, " seconds")

    return resp_obj
