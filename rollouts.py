from collections import namedtuple

import tensorflow as tf

nest = tf.contrib.framework.nest

EnvOutput = namedtuple('EnvOutput', 'reward')


def rollout(env, agent, unroll_length):
    init_env_output = EnvOutput(env.reset())
    init_agent_state = agent.init_state()
    init_agent_output = agent.build(init_agent_state)
    first_values = (init_env_output, init_agent_state, init_agent_output)

    def step(input_, unused_i):
        prev_env_output, prev_agent_state, prev_agent_output = input_
        # reward of 0, random action, cell zero state

        agent_state = agent.update_state(
            prev_env_output, prev_agent_output, prev_agent_state)

        agent_output = agent.build(agent_state)

        env_output = EnvOutput(env.step(agent_output.action))

        return env_output, agent_state, agent_output

    with tf.control_dependencies(nest.flatten(first_values)):
        outputs = tf.scan(
            step,
            tf.range(unroll_length),
            initializer=first_values,
            parallel_iterations=1)

        return outputs
