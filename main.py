import os
import tensorflow as tf

from bandits import Indep, Corr2Arm, ElevenArm
from agent import BanditAgent
from network import BanditNetwork


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_float("gamma", 0.8, "Discount rate")
flags.DEFINE_float("beta_v", 0.05, "Value loss coefficient")
flags.DEFINE_float("beta_e", 0.05, "Entropy loss coefficient")
flags.DEFINE_integer("hidden_units", 48, "LSTM size")

flags.DEFINE_boolean("training", False, "True for training, False for inference")
flags.DEFINE_integer("task", 2, "Task index, 1-based")
flags.DEFINE_string("train_difficulty", 'medium', "Training difficulty for Task 2")
flags.DEFINE_string("test_difficulty", 'easy', "Testing difficulty for Task 2")

flags.DEFINE_integer("train_eps", 30000, "Task index")
flags.DEFINE_integer("test_eps", 150, "Task index")

def main(_):
    if FLAGS.task == 1:
        bandit = Indep(num_arms=2, T=100)
    elif FLAGS.task == 2:
        if FLAGS.training:
            bandit = Corr2Arm(100, FLAGS.train_difficulty)
        else: bandit = Corr2Arm(100, FLAGS.test_difficulty)
    elif FLAGS.task == 3:
        bandit = ElevenArm(5)

    model_directory = './models/task{}/'.format(FLAGS.task) if FLAGS.task != 2 \
        else './models/task{}_{}/'.format(FLAGS.task, FLAGS.train_difficulty)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    tf.reset_default_graph()
    sess = tf.Session()

    net = BanditNetwork(task=FLAGS.task,
                        number_of_actions=bandit.k,
                        hidden_units=FLAGS.hidden_units,
                        learning_rate=FLAGS.learning_rate,
                        beta_v=FLAGS.beta_v,
                        beta_e=FLAGS.beta_e)
    sess.run(tf.global_variables_initializer())

    rlagent = BanditAgent(task=FLAGS.task,
                          number_of_actions=bandit.k,
                          sess=sess,
                          net=net)

    if FLAGS.training:
        rlagent.train(bandit=bandit,
                      num_episodes=FLAGS.train_eps,
                      gamma=FLAGS.gamma,
                      model_directory=model_directory)
    else:
        avg_cumulative_regret = rlagent.test(bandit=bandit,
                                             test_eps=FLAGS.test_eps,
                                             model_directory=model_directory)
        return avg_cumulative_regret[-1]

if __name__ == '__main__':
    tf.app.run()
