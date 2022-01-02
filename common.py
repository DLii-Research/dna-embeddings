import numpy as np
import os
import re
import shelve
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import settransformer as sf
import tf_utils as tfu

i_to_base = [''.join(v) for v in 'NACGT']
base_to_i = { v:i for i, v in enumerate(i_to_base) }

def encode_sequence(sequence):
    return np.array([base_to_i[b] for b in sequence])


def decode_sequence(sequence):
    return ''.join(i_to_base[i] for i in sequence)


def find_shelves(path, suffix="", prepend_path=False):
    files = set([os.path.splitext(f)[0] for f in os.listdir(path) if re.match(r'.*\.(?:db|dat)$', f)])
    if prepend_path:
        return [os.path.join(path, os.path.splitext(f)[0]) for f in files]
    return list(files)


def augment_dna_sequence(sequence, length):
    offset = np.random.randint(0, sequence.shape[1] - length + 1)
    return sequence[:,offset:length+offset]


def encode_kmer_sequence_batch(batch, kmers=3, length=150):
    kmer_powers = (5*tf.ones(kmers, dtype=tf.int32))**tf.range(kmers - 1, -1, -1)
    return tf.reduce_sum(tf.reshape(batch, (batch.shape[0], length//kmers, kmers))*kmer_powers, axis=2)


class DnaSequenceGenerator(tfu.data.MultiprocessDataGenerator):
    def __init__(self, sample_files, length=150, include_quality_scores=True, *args, **kwargs):
        super(DnaSequenceGenerator, self).__init__(*args, **kwargs)
        self.config["sample_files"] = sample_files
        self.config["length"] = length
        self.config["include_quality_scores"] = include_quality_scores
        
    def data_signature(self):
        m = 1 + self.config["include_quality_scores"]
        output_shapes = m*([self.config["length"]],)
        output_types = m*(np.int32,)
        return output_shapes, output_types
    
    @staticmethod
    def generate_batch(data, batch_size, config, global_config):
        store_ids = config["rng"].choice(len(config["stores"]), batch_size)
        batch = np.empty((batch_size, 2, global_config["length"]))
        for i, store_id in enumerate(store_ids):
            store = config["stores"][store_id]
            key = str(np.random.randint(0, len(store)))
            batch[i] = augment_dna_sequence(store[key], global_config["length"])
        if global_config["include_quality_scores"]:
            return (batch[:,0,:], batch[:,1,:])
        return (batch[:,0,:],)
    
    @staticmethod
    def worker_init(config, global_config):
        stores = []
        for sample_file in global_config["sample_files"]:
            store = shelve.open(sample_file)
            stores.append(store)
        config["stores"] = stores
        config["rng"] = np.random.default_rng()
        
    @staticmethod
    def worker_clean(config, global_config):
        for store in config["stores"]:
            store.close()
        del config["stores"]
        
        
class DnaSampleGenerator(DnaSequenceGenerator):
    def __init__(self, sample_files, subsample_size=1000, *args, **kwargs):
        super(DnaSampleGenerator, self).__init__(sample_files, *args, **kwargs)
        self.config["subsample_size"] = subsample_size
        
    def data_signature(self):
        output_shapes = (
            [self.config["subsample_size"], self.config["length"]],
            ())
        output_types = (np.int32, np.int32)
        return output_shapes, output_types
    
    @staticmethod
    def generate_batch(data, batch_size, config, global_config):
        rng = config["rng"]
        store_ids = rng.integers(0, len(config["stores"]), batch_size)
        samples = np.empty((batch_size, global_config["subsample_size"], global_config["length"]))
        for sample_index, store_id in enumerate(store_ids):
            store = config["stores"][store_id]
            keys = map(str, rng.choice(len(store), global_config["subsample_size"], replace=False))
            for sequence_index, key in enumerate(keys):
                samples[sample_index][sequence_index] = augment_dna_sequence(store[key], global_config["length"])[0]
        return (samples, store_ids)
        
        
class BaseTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, ff_activation="gelu", dropout_rate=0.1, prenorm=False):
        super(BaseTransformerBlock, self).__init__()
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.att = self.create_attention_layer(embed_dim, num_heads)
        
        if prenorm:
            self.self_att = self.self_att_prenorm
        else:
            self.self_att = self.self_att_postnorm
        
    def create_attention_layer(self, embed_dim, num_heads):
        raise NotImplemented()
        
    def self_att_prenorm(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm, inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = inputs + attn_output
        
        ffn_norm = self.layernorm2(attn_output)
        ffn_output = self.ffn(ffn_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return attn_output + ffn_output
    
    def self_att_postnorm(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def call(self, inputs, training):
        return self.self_att(inputs, training)
    

class TransformerBlock(BaseTransformerBlock):
    def create_attention_layer(self, embed_dim, num_heads):
        return keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    
    
class FastformerBlock(BaseTransformerBlock):        
    def create_attention_layer(self, embed_dim, num_heads):
        return Fastformer(embed_dim, num_heads)
    
    
class FixedPositionEmbedding(keras.layers.Layer):
    def __init__(self, length, embed_dim):
        super(FixedPositionEmbedding, self).__init__()
        self.length = length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.positions = self.add_weight(
            shape=(self.length, self.embed_dim),
            initializer="uniform",
            trainable=True)
        
    def call(self, x):
        return x + self.positions
    
    
class ConditionedISAB(keras.layers.Layer):
    def __init__(self, embed_dim, dim_cond, num_heads, num_anchors):
        super(ConditionedISAB, self).__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.mab1 = sf.MAB(embed_dim, num_heads)
        self.mab2 = sf.MAB(embed_dim, num_heads)
        self.anchor_predict = keras.models.Sequential([
            keras.layers.Dense(
                2*dim_cond,
                input_shape=(dim_cond,),
                activation="gelu"),
            tfa.layers.SpectralNormalization(
                keras.layers.Dense(num_anchors*embed_dim)),
            keras.layers.Reshape((num_anchors, embed_dim))
        ])
        
    def call(self, inp):
        inducing_points = self.anchor_predict(inp[1])
        h = self.mab1(inducing_points, inp[0])
        return self.mab2(inp[0], h)
    
    
class SampleSet(keras.layers.Layer):
    def __init__(self, max_set_size, embed_dim):
        super(SampleSet, self).__init__()
        self.max_set_size = max_set_size
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.mu = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="mu")
        self.sigma = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="sigma")
        
    def call(self, n):
        batch_size = tf.shape(n)[0]
        n = tf.squeeze(tf.cast(n[0], dtype=tf.int32)) # all n should be the same, take one
        mean = self.mu
        variance = tf.square(self.sigma)
        
        # Sample a random initial set of max size
        initial_set = tf.random.normal((batch_size, self.max_set_size, self.embed_dim), mean, variance)

        # Pick random indices without replacement
        _, random_indices = tf.nn.top_k(tf.random.uniform(shape=(batch_size, self.max_set_size)), n)        
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), n), (-1, n))
        indices = tf.stack([batch_indices, random_indices], axis=2)
        
        # Sample the set
        sampled_set = tf.gather_nd(initial_set, indices)
        
        return sampled_set