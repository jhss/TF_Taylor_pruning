import tensorflow as tf

class GateLayer(tf.keras.layers.Layer):
    def __init__(self, input_features, output_features):
        super(GateLayer, self).__init__()
        self.weight = self.add_weight(shape = (1, 1, 1, input_features),
                                      initializer = tf.keras.initializers.Ones())
        
        self.trainable = False
        
    def call(self, x):
        return tf.multiply(x, self.weight)


if __name__ == '__main__':
    
    x = tf.random.normal(shape = (2, 3, 3, 4))
    layer = GateLayer(4, 4)
    result = layer(x)
