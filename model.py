import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


from tensorflow.keras import layers

# base_encoder
def create_encoder(input_shape):
    mobilenet = MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet',
        input_tensor=None
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = mobilenet(inputs)
    x = layers.Conv2D(512, kernel_size=1)(x)
    outputs = layers.GlobalAvgPool2D()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="base-encoder")
    return model


# projection_head
def add_projection_head(encoder, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    x = layers.Dense(256, activation="relu")(features)  # here it takes 1280 vec_length but in paper it takes 512
    x = layers.Dense(128, activation=None)(x)
    outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="base-encoder_with_projection-head"
    )
    return model

# Loss Function
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        #         tf.print(K.shape(feature_vectors))

        # separate normal and anomaly examples
        n_idxs = tf.reshape(tf.where(labels == 0)[:, 0], [-1, 1])
        a_idxs = tf.reshape(tf.where(labels == 1)[:, 0], [-1, 1])

        n_vectors = tf.gather_nd(feature_vectors, n_idxs)
        a_vectors = tf.gather_nd(feature_vectors, a_idxs)
        # Compute logits
        n_scores = tf.divide(
            tf.matmul(n_vectors, tf.transpose(n_vectors)),
            self.temperature,
        )

        a_n_scores = tf.divide(
            tf.matmul(n_vectors, tf.transpose(a_vectors)),
            self.temperature,
        )
        pos_logits = tf.exp(n_scores)
        neg_logits = tf.exp(a_n_scores)

        # compute loss
        denominator = pos_logits + tf.reduce_sum(neg_logits, axis=-1, keepdims=True)
        loss_steps = -1 * tf.math.log((pos_logits / denominator))
        loss_steps = tf.linalg.set_diag(loss_steps,
                                        tf.zeros(K.shape(loss_steps)[0]))  # remove values for i==j(diagonal)

        k = tf.cast(K.shape(n_scores), tf.float32)[0]
        k = tf.math.maximum(k, 2)  # prevent divide by zero, when the batch contains no or one normal photos
        loss = (1 / (k * (k - 1))) * tf.reduce_sum(loss_steps)
        return loss


def get_train_model(input_shape):
    """
    # Arguments
        optmizer:
        input_shape: 3D shape
    return:
    """
    base_encoder = create_encoder(input_shape)

    encoder_with_projection_head = add_projection_head(base_encoder, input_shape)

    return base_encoder, encoder_with_projection_head


# is this ok since layers.Lambda has no weights, so no need for fitting?
def get_test_model(encoder, input_shape):
    """
    # Arguments
        optmizer:
        input_shape: 3D shape
    return:
    """
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="test_model"
    )
    return model
