import tensorflow as tf
import matplotlib.pyplot as plt
import os

from bandits import Bandits
from agent import BanditAgent

tf.app.flags.DEFINE_boolean("training", False, "True for training, False for inference")
tf.app.flags.DEFINE_integer("task", 2, "Task index, 1-based")
tf.app.flags.DEFINE_string("train_difficulty", 'easy', "Training difficulty for Task 2")
tf.app.flags.DEFINE_string("test_difficulty", 'easy', "Testing difficulty for Task 2")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
tf.app.flags.DEFINE_float("gamma", 0.8, "Discount rate")
tf.app.flags.DEFINE_integer("train_eps", 30000, "Task index")
tf.app.flags.DEFINE_integer("test_eps", 150, "Task index")
tf.app.flags.DEFINE_integer("save_every", 2000, "Save model every x episodes")
FLAGS = tf.app.flags.FLAGS

model_directory = './models/task{}/'.format(FLAGS.task) if FLAGS.task!=2 \
    else './models/task{}_{}_{}/'.format(FLAGS.task, FLAGS.train_difficulty, FLAGS.test_difficulty)
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

if FLAGS.training:
    bandit = Bandits(task = FLAGS.task, difficulty = FLAGS.train_difficulty)
else:
    bandit = Bandits(task = FLAGS.task, difficulty = FLAGS.test_difficulty)

tf.reset_default_graph()
sess = tf.Session()
rlagent = BanditAgent(task = FLAGS.task, sess = sess, learning_rate = FLAGS.learning_rate, number_of_actions = bandit.number_of_actions, number_of_episodes = FLAGS.train_eps)
sess.run(tf.global_variables_initializer())

if FLAGS.training:
    rlagent.train(bandit, FLAGS.train_eps, True, FLAGS.gamma, save_every=FLAGS.save_every, model_directory=model_directory)
else:
    avg_cumulative_regret = rlagent.test(bandit = bandit, test_eps = FLAGS.test_eps, restore = True, model_directory=model_directory)
    plt.plot(avg_cumulative_regret)
    plt.show()
