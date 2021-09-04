import tensorflow as tf
import numpy as np
from tensorflow.keras import models, optimizers, layers, losses

class PhaseLayer(layers.Layer):
    def __init__(self, rng=np.random.RandomState(23456),units=512, input_dim=512, number_of_phases=4):
        super(PhaseLayer, self).__init__()
        self.nslices = number_of_phases
        self.units = units
        self.input_dim = input_dim
        self.rng = rng
    
        self.w = tf.Variable(self.initial_alpha(), name="w", trainable=True)
        self.b = tf.Variable(self.initial_beta(), name="b", trainable=True)
    
    def initial_alpha(self):
        shape = (self.nslices, self.input_dim, self.units)
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            self.rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32
        )
        return tf.convert_to_tensor(alpha, dtype=tf.float32)
    
    def initial_beta(self):
        return tf.zeros((self.nslices, self.units), dtype=tf.float32)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class PFNN(tf.keras.Model):
    def __init__(self, input_dim=1, output_dim=1, dropout=0.3, **kwargs):
        super(PFNN ,self).__init__(**kwargs)
        self.nslices = 4
        self.input_dim=input_dim
        self.output_dim=output_dim
    
        self.dropout0 = layers.Dropout(dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
        self.activation = layers.ELU()
    
        self.layer0 = PhaseLayer(input_dim=input_dim)
        self.layer1 = PhaseLayer()
        self.layer2 = PhaseLayer(units=output_dim)

    def call(self, inputs):
        pscale = self.nslices * inputs[:,-1]
        pamount = pscale % 1.0
    
        pindex_1 = tf.cast(pscale, 'int32') % self.nslices
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices
    
        bamount = tf.expand_dims(pamount, 1)
        wamount = tf.expand_dims(bamount, 1)

        def cubic(y0, y1, y2, y3, mu):
            return (
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
            (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
            (-0.5*y0+0.5*y2)*mu +
            (y1))
    
        W0 = cubic(
            tf.nn.embedding_lookup(self.layer0.w, pindex_0), 
            tf.nn.embedding_lookup(self.layer0.w, pindex_1), 
            tf.nn.embedding_lookup(self.layer0.w, pindex_2), 
            tf.nn.embedding_lookup(self.layer0.w, pindex_3), 
            wamount)
        W1 = cubic(
            tf.nn.embedding_lookup(self.layer1.w, pindex_0), 
            tf.nn.embedding_lookup(self.layer1.w, pindex_1), 
            tf.nn.embedding_lookup(self.layer1.w, pindex_2), 
            tf.nn.embedding_lookup(self.layer1.w, pindex_3), 
            wamount)
        W2 = cubic(
            tf.nn.embedding_lookup(self.layer2.w, pindex_0), 
            tf.nn.embedding_lookup(self.layer2.w, pindex_1), 
            tf.nn.embedding_lookup(self.layer2.w, pindex_2), 
            tf.nn.embedding_lookup(self.layer2.w, pindex_3), 
            wamount)
        
        b0 = cubic(
            tf.nn.embedding_lookup(self.layer0.b, pindex_0), 
            tf.nn.embedding_lookup(self.layer0.b, pindex_1), 
            tf.nn.embedding_lookup(self.layer0.b, pindex_2), 
            tf.nn.embedding_lookup(self.layer0.b, pindex_3), 
            bamount)
        b1 = cubic(
            tf.nn.embedding_lookup(self.layer1.b, pindex_0),
            tf.nn.embedding_lookup(self.layer1.b, pindex_1),
            tf.nn.embedding_lookup(self.layer1.b, pindex_2),
            tf.nn.embedding_lookup(self.layer1.b, pindex_3),
            bamount)
        b2 = cubic(
            tf.nn.embedding_lookup(self.layer2.b, pindex_0),
            tf.nn.embedding_lookup(self.layer2.b, pindex_1),
            tf.nn.embedding_lookup(self.layer2.b, pindex_2),
            tf.nn.embedding_lookup(self.layer2.b, pindex_3),
            bamount)
        
        H0 = inputs[:, :-1]
        H1 = self.activation(tf.matmul(self.dropout0(H0), W0) + b0)
        H2 = self.activation(tf.matmul(self.dropout0(H1), W1) + b1)
        H3 = tf.matmul(self.dropout2(H2), W2) + b2
    
        return H3
    
    def save_checkpoint(self, direction):
        W0 = self.layer0.w.numpy()
        W1 = self.layer1.w.numpy()
        W2 = self.layer2.w.numpy()
        
        b0 = self.layer0.b.numpy()
        b1 = self.layer1.b.numpy()
        b2 = self.layer2.b.numpy()
        np.savez_compressed(direction + "layer0", weights=W0, bias=b0)
        np.savez_compressed(direction + "layer1", weights=W1, bias=b1)
        np.savez_compressed(direction + "layer2", weights=W2, bias=b2)
    def load_checkpoint(self, direction):
        layer0 = np.load(direction + "layer0.npz")
        self.layer0.w.assign(layer0["weights"], name="w")
        self.layer0.b.assign(layer0["bias"], name="b")
        del layer0
        layer1 = np.load(direction + "layer1.npz")
        self.layer1.w.assign(layer1["weights"], name="w")
        self.layer1.b.assign(layer1["bias"], name="b")
        del layer1
        layer2 = np.load(direction + "layer2.npz")
        self.layer2.w.assign(layer2["weights"], name="w")
        self.layer2.b.assign(layer2["bias"], name="b")
        del layer2