#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None)

train_df = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
sub_df = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")

train_df.head()


# In[ ]:


train_df.drop(["Soil_Type7", "Id", "Soil_Type15"], axis=1, inplace=True)
test_df.drop(["Soil_Type7", "Id", "Soil_Type15"], axis=1, inplace=True)


# In[ ]:


train_df = train_df[train_df.Cover_Type != 5]


# In[ ]:


new_names = {
    "Horizontal_Distance_To_Hydrology": "x_dist_hydrlgy",
    "Vertical_Distance_To_Hydrology": "y_dist_hydrlgy",
    "Horizontal_Distance_To_Roadways": "x_dist_rdwys",
    "Horizontal_Distance_To_Fire_Points": "x_dist_firepts"
}

train_df.rename(new_names, axis=1, inplace=True)
test_df.rename(new_names, axis=1, inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
train_df["Cover_Type"] = encoder.fit_transform(train_df["Cover_Type"])


# In[ ]:


train_df["Aspect"][train_df["Aspect"] < 0] += 360
train_df["Aspect"][train_df["Aspect"] > 359] -= 360

test_df["Aspect"][test_df["Aspect"] < 0] += 360
test_df["Aspect"][test_df["Aspect"] > 359] -= 360


# In[ ]:


# Manhhattan distance to Hydrology
train_df["mnhttn_dist_hydrlgy"] = np.abs(train_df["x_dist_hydrlgy"]) + np.abs(train_df["y_dist_hydrlgy"])
test_df["mnhttn_dist_hydrlgy"] = np.abs(test_df["x_dist_hydrlgy"]) + np.abs(test_df["y_dist_hydrlgy"])

# Euclidean distance to Hydrology
train_df["ecldn_dist_hydrlgy"] = (train_df["x_dist_hydrlgy"]**2 + train_df["y_dist_hydrlgy"]**2)**0.5
test_df["ecldn_dist_hydrlgy"] = (test_df["x_dist_hydrlgy"]**2 + test_df["y_dist_hydrlgy"]**2)**0.5


# In[ ]:


train_df.loc[train_df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
test_df.loc[test_df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0

train_df.loc[train_df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
test_df.loc[test_df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0

train_df.loc[train_df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
test_df.loc[test_df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

train_df.loc[train_df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
test_df.loc[test_df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255

train_df.loc[train_df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
test_df.loc[test_df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

train_df.loc[train_df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
test_df.loc[test_df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255


# In[ ]:


features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
soil_features = [x for x in train_df.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in train_df.columns if x.startswith("Wilderness_Area")]

def addFeature(X):
    # Thanks @mpwolke : https://www.kaggle.com/mpwolke/tooezy-where-are-you-no-camping-here
    X["Soil_Count"] = X[soil_features].apply(sum, axis=1)

    # Thanks @yannbarthelemy : https://www.kaggle.com/yannbarthelemy/tps-december-first-simple-feature-engineering
    X["Wilderness_Area_Count"] = X[wilderness_features].apply(sum, axis=1)
    X["Hillshade_mean"] = X[features_Hillshade].mean(axis=1)
    X['amp_Hillshade'] = X[features_Hillshade].max(axis=1) - X[features_Hillshade].min(axis=1)


# In[ ]:


addFeature(train_df)
addFeature(test_df)


# In[ ]:


from sklearn.preprocessing import RobustScaler


cols = [
    "Elevation",
    "Aspect",
    "mnhttn_dist_hydrlgy",
    "ecldn_dist_hydrlgy",
    "Slope",
    "x_dist_hydrlgy",
    "y_dist_hydrlgy",
    "x_dist_rdwys",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "x_dist_firepts",
    
    "Soil_Count","Wilderness_Area_Count","Hillshade_mean","amp_Hillshade"
]

scaler = RobustScaler()
train_df[cols] = scaler.fit_transform(train_df[cols])
test_df[cols] = scaler.transform(test_df[cols])


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# In[ ]:


train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


INPUT_SHAPE = test_df.shape[1:]
NUM_CLASSES = train_df["Cover_Type"].nunique()

def build_model():
    model = Sequential([
        Dense(units=300, kernel_initializer="lecun_normal", activation="selu", input_shape=INPUT_SHAPE),
        BatchNormalization(),
        Dense(units=200, kernel_initializer="lecun_normal", activation="selu"),
        BatchNormalization(),
        Dense(units=100, kernel_initializer="lecun_normal", activation="selu"),
        BatchNormalization(),
        Dense(units=50, kernel_initializer="lecun_normal", activation="selu"),
        BatchNormalization(),
        Dense(units=NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=20,
    restore_best_weights=True
)

callbacks = [reduce_lr, early_stop]


# In[ ]:


from tensorflow.keras.utils import plot_model


plot_model(
    build_model(),
    show_shapes=True,
    show_layer_names=True
)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


X = train_df.drop("Cover_Type", axis=1).values
y = train_df["Cover_Type"].values

del train_df

FOLDS = 20
EPOCHS = 200
BATCH_SIZE = 2048

test_preds = np.zeros((1, 1))
scores = []

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=False
    )

    y_pred = np.argmax(model.predict(X_val), axis=1)
    score = accuracy_score(y_val, y_pred)
    scores.append(score)

    test_preds = test_preds + model.predict(test_df)
    print(f"Fold {fold} Accuracy: {score}")

print()
print(f"Mean Accuracy: {np.mean(scores)}")


# In[ ]:


test_preds = np.argmax(test_preds, axis=1)
test_preds = encoder.inverse_transform(test_preds)

sub_df['Cover_Type'] = test_preds
sub_df.head()


# In[ ]:


sub_df.to_csv("submission.csv", index=False)

