import tensorflow as tf
import numpy as np

from env import TFBandit
from agents import L2RLAgent
from worker import Worker


nest = tf.contrib.framework.nest
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer("rnn_size", 48, "LSTM size.")
flags.DEFINE_integer("unroll_length", 100, "Number of trials.")
flags.DEFINE_integer("train_steps", 30000, "Number of train episodes.")
flags.DEFINE_integer("test_eps", 150, "Number of test episodes.")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_boolean("eval_mode", True, "True if evaluating performance.")
flags.DEFINE_string("train_difficulty", 'medium', "Training difficulty.")
flags.DEFINE_string("test_difficulty", 'easy', "Testing difficulty.")

# Optimisation.
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("gamma", 0.8, "Discount rate.")
flags.DEFINE_float("beta_v", 0.05, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.05, "Entropy loss coefficient.")
flags.DEFINE_float("rms_decay", 0.99, "RMS decay.")

# Model saving.
flags.DEFINE_string("model_directory", './models/', "Model directory.")
flags.DEFINE_boolean("save_model", False, "True if saving model.")
flags.DEFINE_boolean("load_model", False, "True if loading model.")

DIFFICULTIES = {
    'easy': [0.1, 0.9],
    'medium': [0.25, 0.75],
    'hard': [0.4, 0.6]
    }


def set_random_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def train(constants):
    set_random_seeds(constants.seed)

    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=constants.learning_rate, decay=constants.rms_decay)

    env = TFBandit(DIFFICULTIES[constants.train_difficulty])
    agent = L2RLAgent(constants.rnn_size, env.action_space)

    worker = Worker(env, agent, constants)
    worker.build_rollout()
    worker.build_loss()
    worker.build_optimisation(optimiser)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    vars_to_save_load = tf.trainable_variables()
    if constants.load_model or constants.save_model:
        saver = tf.train.Saver(vars_to_save_load)
    if constants.load_model:
        worker.load_model(sess, saver, constants.model_directory)

    worker.train(sess)

    # Save shared variables.
    if constants.save_model:
        worker.save_model(sess, saver, constants.model_directory)
    sess.close()


def evaluate(constants):
    env = TFBandit(DIFFICULTIES[constants.test_difficulty])
    agent = L2RLAgent(constants.rnn_size, env.action_space)
    worker = Worker(env, agent, constants)
    worker.build_rollout()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    if FLAGS.load_model:
        vars_to_save_load = tf.trainable_variables()
        saver = tf.train.Saver(vars_to_save_load)
        worker.load_model(sess, saver, constants.model_directory)
    worker.evaluate(sess)
    sess.close()


def main(_):
    if FLAGS.eval_mode:
        evaluate(FLAGS)
    else:
        train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
