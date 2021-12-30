import pandas as pd

from model import get_train_model, SupervisedContrastiveLoss, get_test_model
from perpare_dataset import get_data_schema, load_train_data, load_test_data, load_normal_data
import tensorflow as tf

from utils import get_normal_vector, MakePrediction

if __name__ == '__main__':
    learning_rate = 0.01  # try and error
    temperature = 0.1 # try and error
    batch_size = 32
    input_shape = (112, 112, 3)  # try and error
    threshold = 0.51 # try and error

    # =================================Loading Data=================================
    #create train_ds schema
    # get_data_schema("C:\\Users\\saraz\\ITI\\Graduation\\Camera 2\\train", "c0", "AUC_train_schema")

    # create test_ds schema
    # get_data_schema("C:\\Users\\saraz\\ITI\\Graduation\\Camera 2\\test", "c0", "AUC_test_schema")

    train_ds = load_train_data("AUC_train_schema", batch_size, input_shape[:2])

    validation_ds, test_ds = load_test_data("AUC_test_schema", batch_size, input_shape[:2])

    # =================================Training Model=================================

    encoder, train_model = get_train_model(input_shape)

    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay( #not sure
    #     initial_learning_rate=0.01,
    #     decay_steps=100,
    #     decay_rate=0.1)
    train_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate,
                                          # momentum=0.9,
                                          ),
        loss=SupervisedContrastiveLoss(temperature)
    )
    history = train_model.fit(train_ds, epochs=20)
    print(history.history)

    # Get the normal vactor & test_model
    test_model = get_test_model(encoder, input_shape)

    normal_vector = get_normal_vector("AUC_train_schema", batch_size, input_shape[:2], test_model)

    # =================================Validate And Test=================================

    predict_obj = MakePrediction(test_model, normal_vector, threshold)

    y_valid_pred = predict_obj.predict(validation_ds)
    y_test_pred = predict_obj.predict(test_ds)

    y_true = pd.read_csv("AUC_test_schema.csv")["Label"]
    y_test_true, y_valid_true = y_true[:400], y_true[400:]

    # Get AUC
    m = tf.keras.metrics.AUC()
    m.update_state(y_valid_true, y_valid_pred)
    valid_auc = m.result().numpy()
    print(f"validation AUC = {valid_auc}")

    m.update_state(y_test_true, y_test_pred)
    test_auc = m.result().numpy()
    print(f"test AUC = {test_auc}")





