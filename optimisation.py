import tensorflow as tf

from losses import advantage_loss, compute_return, policy_gradient, entropy

nest = tf.contrib.framework.nest


def policy_loss_calc(logits, actions, advantages, constants):
    policy_loss = policy_gradient(logits, actions, advantages)
    entropy_loss = constants.beta_e * entropy(logits)
    return policy_loss + entropy_loss


def advantage_calc(values, rewards, discounts):
    true_returns = compute_return(rewards, discounts)
    return true_returns - values


def reward_fn(env_rewards, *args):
    rewards = env_rewards
    rewards = tf.clip_by_value(rewards, -1.0, 1.0)
    return rewards


def loss_calculation(rollout_outputs, action_space, constants):
    env_outputs, _, agent_outputs = rollout_outputs
    env_outputs, agent_outputs = nest.map_structure(
        lambda t: tf.squeeze(t), (env_outputs, agent_outputs))

    # Subset all.
    env_outputs, agent_outputs = nest.map_structure(
        lambda t: t[:-1], (env_outputs, agent_outputs))

    env_rewards = env_outputs.reward
    rewards = reward_fn(env_rewards)

    gamma = tf.constant(constants.gamma, tf.float32)
    discounts = tf.ones_like(agent_outputs.values) * gamma

    advantages = advantage_calc(agent_outputs.values, rewards, discounts)
    return_loss = constants.beta_v * advantage_loss(advantages)

    policy_loss = policy_loss_calc(
        agent_outputs.logits, agent_outputs.action, advantages, constants)
    return return_loss + policy_loss
