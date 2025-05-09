"""Minimal NN helpers: build, train, save."""

import os, pickle
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --------------------------------------------------
#  Build simple MLP model
# --------------------------------------------------

def create_nn_model(
    user_vocab_size,
    numerical_feature_size,
    embedding_dim,
    user_embedding_dim = 8,
    dropout_rate = 0.3,
    l2_reg = 0.001,
    dense_layers_config = [64, 32]
):
    """Return a compiled Keras model."""
    user_in   = layers.Input(shape=(1,), name="user_input")
    num_in    = layers.Input(shape=(numerical_feature_size,), name="numerical_input")
    inputs = [user_in, num_in]

    user_vec  = layers.Embedding(user_vocab_size, user_embedding_dim)(user_in)
    user_vec  = layers.Flatten()(user_vec)
    user_vec  = layers.Dropout(dropout_rate)(user_vec)

    num_vec   = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(num_in)
    num_vec   = layers.Dropout(dropout_rate)(num_vec)

    feats = [user_vec, num_vec]

    if embedding_dim > 0:
        emb_in  = layers.Input(shape=(embedding_dim,), name="embedding_input")
        inputs.append(emb_in)
        emb_vec = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(emb_in)
        emb_vec = layers.Dropout(dropout_rate)(emb_vec)
        feats.append(emb_vec)

    x = layers.Concatenate()(feats)
    for units in dense_layers_config:
        x = layers.Dense(units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

# --------------------------------------------------
#  Train with early stopping
# --------------------------------------------------

def train_nn_model(
    model,
    train_ds,
    train_y,
    val_ds,
    val_y,
    epochs = 50,
    batch_size = 1024,
    patience_es = 7
):
    early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        train_ds,
        train_y,
        validation_data=(val_ds, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=2,
    )
    return history

# --------------------------------------------------
#  Save final model
# --------------------------------------------------

def save_nn_model(model, results_dir, filename="final_nn_model.keras"):
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    model.save(os.path.join(results_dir, "models", filename))
