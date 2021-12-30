import os
import pandas as pd
import glob

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from custom_data_generator.custom_image_data_generator import CustomImageDataGenerator


def get_data_schema(dir_name, normal_str, csv_file_name, view=""):
    """
    generate csv file contains two columns:
        "Path": the path for each image
        "Label": the label of each image

    # Arguments
        dir_name: path for the dataset directory which contains the classes folders
        normal_str: the starting string of the normal data folders
        csv_file_name: Name of the generated csv file
        view: One of "front_depth", "front_IR", "top_depth", "top_IR" or "" if we use AUC dataset
    """
    if view:  # if we use DAD dataset
        testers = [folder for folder in os.listdir(f"{dir_name}/") if folder.startswith("Tester")]
        folders_names = os.listdir(dir_name + "/" + testers[0])
    else:  # if we use camera2 dataset
        testers = "/"
        folders_names = os.listdir(dir_name)

    path_lst = []
    label_lst = []
    for tester in testers:
        for folder in folders_names:
            label = 0 if folder.startswith(normal_str) else 1
            paths = glob.glob(f'{dir_name}/{tester}/{folder}/{view}/*')
            labels = [label] * len(paths)

            path_lst.extend(paths)
            label_lst.extend(labels)

    data_tuples = list(zip(path_lst, label_lst))
    df = pd.DataFrame(data_tuples, columns=['Path', 'Label'])

    df.to_csv(f"{csv_file_name}.csv", header=True, index=False)


def load_train_data(schema_file, batch_size, input_shape):
    """
    # Arguments
        schema_file: path to the csv schema file
        batch_size:
        input_shape: 2D shape

    # return: CustomDataFrameIterator object contains tha train dataset
    """
    train_data_schema = pd.read_csv(f"{schema_file}.csv", dtype=str)

    # sort the dataframe according to Label (all normal photos, then the anomaly), then custom_flow_from_directory would
    # shuffle the data so each epoch contain the same ratio of normal:anomaly
    train_data_schema.sort_values("Label", inplace=True)
    normal_count = (train_data_schema["Label"] == "0").sum()

    datagen = CustomImageDataGenerator(
        rotation_range=20,  # try_and_error
        channel_shift_range=20,  # try_and_error
        horizontal_flip=True,
        preprocessing_function=preprocess_input  # scale input pixels between -1 and 1.
    )
    train_ds = datagen.flow_from_dataframe(
        dataframe=train_data_schema,
        directory=".",
        x_col="Path",
        y_col="Label",
        batch_size=batch_size,
        # seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=input_shape,
        positive_n=normal_count,
    )
    return train_ds


def load_test_data(schema_file, batch_size, input_shape):
    """
    # Arguments
        schema_file: path to the csv schema file
        directory: folder that contain tha dataset, the starting point of the images paths
        batch_size:
        input_shape: 2D shape

    # return: 2 CustomDataFrameIterator objects contain tha validation and test data
    """
    test_data_schema = pd.read_csv(f"{schema_file}.csv", dtype=str)
    # # sort the dataframe according to Label
    # test_data_schema.sort_values("Label", inplace=True)

    datagen = CustomImageDataGenerator(preprocessing_function=preprocess_input,
                                       validation_split=0.5)

    validation_ds = datagen.flow_from_dataframe(
        dataframe=test_data_schema,
        directory=".",
        x_col="Path",
        y_col="Label",
        batch_size=batch_size,
        target_size=input_shape,
        class_mode="categorical",
        shuffle=False,
        subset="validation")

    test_ds = datagen.flow_from_dataframe(
        dataframe=test_data_schema,
        directory=".",
        x_col="Path",
        y_col="Label",
        batch_size=batch_size,
        target_size=input_shape,
        class_mode="categorical",
        shuffle=False,
        subset="training")

    return validation_ds, test_ds


def load_normal_data(schema_file, batch_size, input_shape):
    """
    # Arguments
        schema_file: path to the csv schema file for train data
        batch_size:
        input_shape: 2D shape

    # return: CustomDataFrameIterator object contains tha normal train data
    """

    train_data_schema = pd.read_csv(f"{schema_file}.csv", dtype=str)
    normal_data_schema = train_data_schema.loc[train_data_schema["Label"] == "0"]

    datagen = CustomImageDataGenerator(preprocessing_function=preprocess_input)

    normal_ds = datagen.flow_from_dataframe(
        dataframe=normal_data_schema,
        directory=".",
        x_col="Path",
        y_col="Label",
        class_mode="categorical",
        batch_size=batch_size,
        target_size=input_shape,
        shuffle=False)

    return normal_ds
