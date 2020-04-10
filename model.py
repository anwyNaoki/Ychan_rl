import numpy as np
from tensorflow.python import keras as K
from Experience import Experience

from tensorflow.losses import huber_loss
class Model:
    def __init__(self):
        self._model = None
        self._teacher_model = None

    def predict(self, s):
        return self._model.predict(np.array([s]))[0]

    def update(self, experiences):
        pass

    def save_model(self):
        self._model.save("hogehoge.h5", overwrite=True, include_optimizer=True)


class DQNModel(Model):
    def __init__(self,gamma=0.99,learning_rate=0.0005,ddqn=True):
        super().__init__()
        self.gamma = gamma
        self.ddqn = True
        self.learning_rate=learning_rate

    def predict(self, s):
        return self._model.predict(np.array([s]))[0]


    def check_attention(self, s, name):
        l = self.intermediate_layer_model.predict(np.array([s]))[0]
        print(l)





    def set_model(self):
        normal = K.initializers.glorot_normal()
        l_input = K.layers.Input(shape=(5, 5, 11))  # 16
        l = []
        Dlayer=K.layers.DepthwiseConv2D(5, padding="same", kernel_initializer=normal, activation="relu")
        x = Dlayer(l_input)
        flatten_layer = K.layers.Flatten()
        GAP = K.layers.GlobalAveragePooling2D()
        x = GAP(x)
        x = K.layers.Activation('softmax')(x)
        x = K.layers.multiply([x, l_input])

        x = K.layers.Conv2D(32, 3, strides=1, padding="same", kernel_initializer=normal, activation="relu")(x)
        x = K.layers.Conv2D(32, 3, strides=1, padding="same", kernel_initializer=normal, activation="relu")(x)
        x = flatten_layer(x)
        x = K.layers.Dense(128, activation="linear")(x)
        x = K.layers.Dense(4, activation="linear")(x)
        m = K.models.Model(inputs=l_input, outputs=x)
        self._model = m
        self._teacher_model = K.models.clone_model(self._model)
        optimizer = K.optimizers.Adam(lr=self.learning_rate)
        self._model.compile(optimizer, loss="mse")
        self._model.summary()
        self.intermediate_layer_model = K.models.Model(inputs=l_input, outputs=self._model.get_layer("global_average_pooling2d").output)

    
    
    def reset_teacher(self):
        self._teacher_model.set_weights(self._model.get_weights())

    def update(self, experiences):
     
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        m_estimateds = self._model.predict(states)
        t_estimateds = self._teacher_model.predict(n_states)
        ddqn_estimateds = self._model.predict(n_states)

        
        for i, e in enumerate(experiences):
            if not e.d:
                #DDQN仕様の実装
                m_estimateds[i][e.a] = e.r + self.gamma *  t_estimateds[i][np.argmax(ddqn_estimateds[i])]
            else:
                m_estimateds[i][e.a] = e.r

        loss = self._model.fit(states, m_estimateds,epochs=1,verbose=0)
        return loss


    def return_td(self,experiences,index):
        states = np.array([b.s for b in experiences])
        n_states = np.array([b.n_s for b in experiences])

        m_estimateds = self._model.predict(states)
        t_estimateds = self._teacher_model.predict(n_states)
        ddqn_estimateds = self._model.predict(n_states)

        tds=[]
        for j in range(len(experiences)):
            td = abs(experiences[j].r + (self.gamma) * t_estimateds[j][np.argmax(ddqn_estimateds[j])] - m_estimateds[j][experiences[j].a])
            tds.append(td)
        return tds
