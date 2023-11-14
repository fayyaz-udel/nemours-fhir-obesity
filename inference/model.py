import tensorflow as tf
from keras import metrics
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

tf.compat.v1.disable_eager_execution()
from keras.layers import LSTM, Activation

METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.AUC(name='auc'),
]


def create_model(static_feature_dim, temporal_feature_dim, temporal_steps):
    input_demo = Input(shape=static_feature_dim)
    input_temp = Input(shape=(temporal_steps, temporal_feature_dim))

    lstm = LSTM(units=8, return_sequences=True)(input_temp)
    lstm = LSTM(units=16, activation='tanh')(lstm)

    dense = Dense(8)(input_demo)
    dense = Activation('relu')(dense)
    dense = Dense(16)(dense)
    dense = Activation('relu')(dense)

    concat = Concatenate()([lstm, dense])

    first_point = Dense(32)(concat)
    first_point = Activation('relu')(first_point)
    first_point = Dense(16)(first_point)
    first_point = Activation('relu')(first_point)
    output_first_point = Dense(1, activation='sigmoid', name="first_point")(first_point)

    second_point = Dense(32)(concat)
    second_point = Activation('relu')(second_point)
    second_point = Dense(16)(second_point)
    second_point = Activation('relu')(second_point)
    output_second_point = Dense(1, activation='sigmoid', name="2y_after")(second_point)

    model = Model(inputs=[input_demo, input_temp], outputs=[output_first_point, output_second_point])

    model.compile(optimizer="adam", loss=['binary_crossentropy', 'binary_crossentropy'], metrics=METRICS)

    return model
