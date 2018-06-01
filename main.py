import tensorflow as tf
import matplotlib.pyplot as plt
import os

from bandits import Indep, Corr2Arm, ElevenArm
from agent import BanditAgent

tf.app.flags.DEFINE_boolean("training", False, "True for training, False for inference")
tf.app.flags.DEFINE_integer("task", 2, "Task index, 1-based")
tf.app.flags.DEFINE_string("train_difficulty", 'easy', "Training difficulty for Task 2")
tf.app.flags.DEFINE_string("test_difficulty", 'easy', "Testing difficulty for Task 2")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("gamma", 0.8, "Discount rate")
tf.app.flags.DEFINE_integer("train_eps", 30000, "Task index")
tf.app.flags.DEFINE_integer("test_eps", 150, "Task index")
tf.app.flags.DEFINE_integer("save_every", 2000, "Save model every x episodes")
FLAGS = tf.app.flags.FLAGS

model_directory = './models/task{}/'.format(FLAGS.task) if FLAGS.task!=2 else './models/task{}_{}/'.format(FLAGS.task, FLAGS.train_difficulty)
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

if FLAGS.task == 1:
    bandit = Indep(100)
elif FLAGS.task == 2:
    if FLAGS.training:
        bandit = Corr2Arm(100, FLAGS.train_difficulty)
    else: bandit = Corr2Arm(100, FLAGS.test_difficulty)
else:
    bandit = ElevenArm(5)
    
tf.reset_default_graph()
sess = tf.Session()
rlagent = BanditAgent(task = FLAGS.task, sess = sess, learning_rate = FLAGS.learning_rate, number_of_actions = bandit.k, number_of_episodes = FLAGS.train_eps)
sess.run(tf.global_variables_initializer())

if FLAGS.training:
    rlagent.train(bandit = bandit, num_episodes = FLAGS.train_eps, gamma = FLAGS.gamma, save_progress = True, \
                  save_every=FLAGS.save_every, model_directory=model_directory)
else:
    avg_cumulative_regret = rlagent.test(bandit = bandit, test_eps = FLAGS.test_eps, restore = True, model_directory=model_directory)
    plt.plot(avg_cumulative_regret)
    plt.show()
