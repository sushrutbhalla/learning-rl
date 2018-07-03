import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name, share_top_layer):
        self.name = name
        self.share_top_layer = share_top_layer

    @property
    def vars(self):
        #if self.share_top_layer and ("target" not in self.name and "noise" not in self.name):
        if self.share_top_layer:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared_top_layer") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


    @property
    def trainable_vars(self):
        #if self.share_top_layer and ("target" not in self.name and "noise" not in self.name):
        if self.share_top_layer:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared_top_layer") + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        else:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, share_top_layer=False):
        super(Actor, self).__init__(name=name,share_top_layer=share_top_layer)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        used_shared_layer = False
        x = obs
        #if self.share_top_layer and ("target" not in self.name and "noise" not in self.name):
        if self.share_top_layer:
            with tf.variable_scope("shared_top_layer", reuse=tf.AUTO_REUSE) as shared_scope:
                if reuse:
                    shared_scope.reuse_variables()
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                used_shared_layer = True
                print ("------------------------------------------------- sharing layers: {}".format(self.name))

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            #moving obs out of scope. don't think this will change the graph
            #x = obs

            #if we are not sharing the top layer, then this will keep the original layout
            if not used_shared_layer:
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                print ("--------------------------------------------- NOT sharing layers: {}".format(self.name))

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, share_top_layer=False):
        super(Critic, self).__init__(name=name,share_top_layer=share_top_layer)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False, reuse_shared_layer=False):
        x = obs
        used_shared_layer = False
        #if self.share_top_layer and ("target" not in self.name and "noise" not in self.name):
        if self.share_top_layer:
            #the Critic network is created after the actor network, so reuse should be true here
            with tf.variable_scope("shared_top_layer", reuse=True) as shared_scope:
                #if reuse:
                #    shared_scope.reuse_variables()
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                used_shared_layer = True
                print ("------------------------------------------------- sharing layers: {}".format(self.name))

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            #moving obs out of scope. don't think this will change the graph
            #x = obs

            #if we are not sharing the top layer, then this will keep the original layout
            if not used_shared_layer:
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                print ("----------------------------------------------NOT sharing layers: {}".format(self.name))

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

if __name__ == '__main__':
    #validation code to check if sharing of layers works.
    #change value of share_layer to print actor and critic with different layers or shared layers
    import pprint
    share_layer = True
    obs = tf.get_variable("obs", [1,1])
    nb_actions=1
    layer_norm=True
    actions = tf.placeholder(tf.float32, shape=(1,1), name='actions')
    if not share_layer:
        #create a actor network
        actor = Actor(nb_actions, layer_norm=layer_norm, share_top_layer=False)
        actor_tf = actor(obs)
        model = actor

        #create a critic network
        critic = Critic(layer_norm=layer_norm, share_top_layer=False)
        critic_tf = critic(obs, actions)
        model1 = critic

    else:
        #create a actor network with shared_layer
        shared_actor = Actor(nb_actions, layer_norm=layer_norm, share_top_layer=True)
        shared_actor_tf = shared_actor(obs)
        model = shared_actor

        #create a critic network with shared_layer
        shared_critic = Critic(layer_norm=layer_norm, share_top_layer=True)
        shared_critic_tf = shared_critic(obs, actions)
        model1 = shared_critic

    #print results
    print ("\n-------------------- [ACTOR]: ")
    print ("vars:")
    pprint.pprint (model.vars)
    print ("trainable_vars:")
    pprint.pprint (model.trainable_vars)
    print ("perturbable_vars:")
    pprint.pprint (model.perturbable_vars)

    print ("\n-------------------- [CRITIC]: ")
    print ("vars:")
    pprint.pprint (model1.vars)
    print ("trainable_vars:")
    pprint.pprint (model1.trainable_vars)
    print ("perturbable_vars:")
    pprint.pprint (model1.perturbable_vars)
    print ("output_vars:")
    pprint.pprint (model1.output_vars)
