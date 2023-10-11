import time
import cv2
import os
import pandas as pd
from matplotlib import pyplot as plt

from deepface.commons import distance, functions
import recognition_backend

def scalef(coeff, width, is_round=True):
    value = coeff * width ** 0.7
    return round(value) if is_round else value

def threshold_to_confidence(distance, threshold):
    return 100 * (1 - distance / threshold)

def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.gca().set_position((0, 0, 1, 1))
    plt.imshow(img)
    plt.show()

def recognize_image(
        img_path,
        representations,
        extended_models=[],
        detector_backend="retinaface", 
        model_name="Facenet512", 
        distance_metric="cosine",
        enforce_detection=True,
        align=True,
        normalization="base", 
        threshold=None, 
        highlight_gender=False,
        show=True, 
        silent=False
    ):
    
    if threshold == None:
        threshold = distance.findThreshold(model_name, distance_metric)
    
    try:
        img = functions.load_image(img_path)
        width, height, _ = img.shape
    except ValueError as e:
        print("Image not found.")
        print(e)
        return
    
    dfs = recognition_backend.find(
        img_path=img, 
        representations=representations,
        extended_models=extended_models,
        enforce_detection=enforce_detection, 
        detector_backend=detector_backend,
        align=align,
        normalization=normalization,
        model_name=model_name, 
        distance_metric=distance_metric, 
        silent=silent
    )

    actions = list(extended_models.keys())
    matches = pd.DataFrame()

    if len(dfs) > 0:

        matches = pd.DataFrame([face.to_dict(orient="records")[0] for face in dfs])

        for index, face in matches.iterrows():
            if highlight_gender and "gender" in actions:
                line_color =  (255, 255, 0) if face["gender"] == "Man" else (255, 0, 255)
            else:
                line_color = (3, 173, 255)
            x, y, w, h = face["source_x"], face["source_y"], face["source_w"], face["source_h"]
            cv2.rectangle(img, (x, y), (x + w, y + h), line_color, round(w / 40))
            cv2.circle(img, (x, y), scalef(1.7, w), line_color, -1)
        
        for index, face in matches.iterrows():
            if highlight_gender and "gender" in actions:
                line_color =  (255, 255, 0) if face["gender"] == "Man" else (255, 0, 255)
            else:
                line_color = (3, 173, 255)
            x, y, w, h = face["source_x"], face["source_y"], face["source_w"], face["source_h"]
            if face[f'{model_name}_{distance_metric}'] < threshold:
                cv2.putText(img, f"{round(threshold_to_confidence(face[f'{model_name}_{distance_metric}'], threshold))}%", (x - scalef(1.3, w), y + scalef(0.35, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.04, w, is_round=False), (0, 0, 0), scalef(0.08, w))
                cv2.putText(img, str(face["name"]), (x + scalef(1.7, w), y - scalef(0.4, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.05, w, is_round=False), (0, 0, 0), scalef(0.18, w))
                cv2.putText(img, str(face["name"]), (x + scalef(1.7, w), y - scalef(0.4, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.05, w, is_round=False), line_color, scalef(0.1, w))
            else:
                cv2.putText(img, "?", (x - scalef(0.7, w), y + scalef(0.8, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.08, w, is_round=False), (0, 0, 0), scalef(0.15, w))
        
        if not silent:
            print(matches)
        if show:
            show_image(img)

    elif show:
        show_image(img)

    return matches, img

def recognize_faces(
        input_path="inputs",
        output_path="outputs",        
        db_path="datasets", 
        detector_backend="retinaface", 
        model_name="Facenet512", 
        distance_metric="cosine",
        enforce_detection=True,
        align=True,
        normalization="base", 
        threshold=None, 
        actions=(), 
        highlight_gender=False,
        show=True, 
        silent=False
    ):

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)

    representations, extended_models = recognition_backend.init(
        db_path = db_path,
        model_name= model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        actions=actions,
        align=align,
        normalization=normalization,
        silent=silent 
    )

    params = {key: value for key, value in locals().items() if key not in ["input_path", "output_path", "db_path", "actions"]}

    accepted_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    labels = [fn for fn in os.listdir(input_path) if fn.split(".")[-1] in accepted_extensions]

    if(len(labels) == 0):
        print("Directory is empty. Please insert some images inside first. Accepted file types: jpg, jpeg, png")
        return
    else:
        print(f"\nProcessing {len(labels)} image(s).\n{'-' * 40}\n")
        all_matches = dict()
        for label in labels:
            if not silent:
                print(f"Processing image {label}")
            matches, annotated_image = recognize_image(os.path.join(input_path, label), **params)
            all_matches[os.path.join(input_path, label)] = matches
            cv2.imwrite(os.path.join(output_path, label), annotated_image)
            print(f"\n{'-' * 40}\n")
    
    print("Done.")
    return all_matches

def stream_video(
        db_path,
        detector_backend="ssd", 
        model_name="Facenet512", 
        distance_metric="cosine",
        enforce_detection=True,
        align=True,
        normalization="base", 
        threshold=None, 
        actions=(), 
        highlight_gender=False, 
        show=False,
        silent=False
    ):

    if not silent:
        print("Initializing video capture...")
    
    kwargs = {key: value for key, value in locals().items() if key not in ["db_path", "actions"]}
    kwargs["silent"] = True

    vid = cv2.VideoCapture(0)
    frames = 0

    representations, extended_models = recognition_backend.init(
        db_path = db_path,
        model_name= model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        actions=actions,
        align=align,
        normalization=normalization,
        silent=True 
    )
    
    while(True):
        
        ret, frame = vid.read()
        
        _, img = recognize_image(frame, representations, extended_models, **kwargs)

        if frames == 0 and not silent:
            tic = time.time()
        cv2.imshow("win", img)

        frames += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not silent:
        toc = time.time()
        print(f"Frames per second: {frames / (toc - tic)}")
    
    vid.release()
    cv2.destroyAllWindows()

all_matches = recognize_faces(
    show=True,
    db_path="./datasets/sample_dataset", 
    input_path="inputs", 
    output_path="outputs", 
    detector_backend="retinaface", 
    enforce_detection=True,
    threshold=1
)

print(all_matches)

# stream_video(
#     threshold=1, 
#     db_path="./datasets/sample_dataset", 
#     detector_backend="ssd",
# )