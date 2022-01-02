import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import wandb

import tf_utils as tfu

import common


def load_3mer_64(weights_path="./models/embed_3mer_64.h5"):
    config = wandb.config = {
        "learning_rate": 1e-3,
        "learning_rate_schedule": 0.95,
        "batch_size": 256,
        "sequence_length": 150,
        "kmers": 3,
        "embed_dim": 64,
        "num_heads": 8,
        "enc_stack": 2,
        "dec_stack": 2,
        "latent_dim": 64,
        "prenorm": True
    }
    return create_model(config, weights_path)


def load_1mer_4(weights_path="./models/embed_1mer_4.h5"):
    config = wandb.config = {
        "learning_rate": 1e-3,
        "learning_rate_schedule": 0.95,
        "batch_size": 256,
        "sequence_length": 9,
        "kmers": 1,
        "embed_dim": 64,
        "num_heads": 8,
        "enc_stack": 2,
        "dec_stack": 2,
        "latent_dim": 4,
        "prenorm": True
    }
    return create_model(config, weights_path)


def create_model(config, weights_path):
    
    kmers = config["kmers"]
    sequence_length = config["sequence_length"]//config["kmers"]
    num_tokens = 5**config["kmers"]
    embed_dim = config["embed_dim"]
    latent_dim = config["latent_dim"]
    enc_stack = config["enc_stack"]
    dec_stack = config["dec_stack"]
    num_heads = config["num_heads"]
    prenorm = config["prenorm"]

    # Encoder
    y = x = keras.layers.Input((sequence_length,))
    y = keras.layers.Embedding(input_dim=num_tokens, output_dim=embed_dim)(y)
    y = common.FixedPositionEmbedding(length=sequence_length, embed_dim=embed_dim)(y)
    for _ in range(enc_stack):
        y = common.TransformerBlock(embed_dim, num_heads, ff_dim=embed_dim, prenorm=prenorm)(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(latent_dim)(y)
    y = keras.layers.LayerNormalization()(y)
    encoder = keras.Model(x, y, name="Encoder")

    # Decoder
    y = x = keras.layers.Input((encoder.output.shape[1:]))
    y = keras.layers.Dense(sequence_length*embed_dim)(y)
    y = keras.layers.Reshape((-1, embed_dim))(y)
    y = embed = common.FixedPositionEmbedding(length=sequence_length, embed_dim=embed_dim)(y)
    for _ in range(dec_stack):
        y = common.TransformerBlock(embed_dim, num_heads, ff_dim=embed_dim, prenorm=prenorm)(y)
    y = keras.layers.Dense(num_tokens, activation="softmax")(y)
    decoder = keras.Model(x, y, name="Decoder")

    # Coupled model
    y = x = keras.layers.Input(encoder.input.shape[1:])
    y = encoder(y)
    y = decoder(y)
    model = keras.Model(x, y, name="Autoencoder")
    model.encoder = encoder
    model.decoder = decoder
    
    model.load_weights(weights_path)
    
    return model
    