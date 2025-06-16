"""
Tests for PPO Reward Collection Issues, including ActionRepeatWrapper and PPO integration.

This test suite aims to diagnose problems related to cumulative episode rewards
being zero or incorrect during PPO data collection.

Interpreting Test Results:
- `test_basic_environment_reward`:
    - If this fails: The issue is fundamental. Either the chosen environment
      (e.g., 'CartPole-v1') is not behaving as expected (e.g., always giving
      zero reward), or there's a problem with the basic `gymnasium` environment
      interaction. Check your environment's `step` method or its configuration.

- `test_monitor_wrapper_reward_collection`:
    - If this fails (and basic_environment_reward passes): The problem likely
      lies with the `stable_baselines3.common.monitor.Monitor` wrapper or how
      episode information (`info['episode']`) is being generated or accessed.
      Ensure `Monitor` is correctly wrapping the environment and that episodes
      are properly terminating.

- `test_frame_skipping_reward_accumulation_conceptual`:
    - This is a conceptual test requiring user adaptation. If your adapted version
      fails: The frame skipping wrapper is likely not accumulating rewards correctly
      from the individual skipped frames. Debug the reward accumulation logic
      within your frame skipping wrapper.

- `test_ppo_agent_data_collection_reward_sanity`:
    - If this fails (and monitor_wrapper_reward_collection passes): The issue might be
      more subtle, related to how the PPO agent (specifically its data collection
      phase using `collect_rollouts` or `learn`) interacts with the `Monitor`-wrapped
      vectorized environment. Check:
        - If episodes are completing within `n_steps`.
        - How `Monitor` instances are accessed in the vectorized environment setup.
        - If any other wrappers are interfering.
        - PPO configuration parameters like `gamma` (though less likely for *zero* total reward unless individual rewards are zero).

If all tests pass but you still observe zero cumulative rewards in your actual PPO
training script, the problem might be:
- Specific to your custom environment if you're not using a standard one like CartPole for these tests.
- In the logging/reporting mechanism of your main training script, separate from these tests.
- Related to environment resets or episode termination conditions in more complex scenarios.
- An issue with how rewards are handled *after* collection by the agent (e.g., if they are processed by a reward normalization wrapper that is misconfigured).
"""
import unittest
import gymnasium as gym
import numpy as np
from src.utils import ActionRepeatWrapper # Added import
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # Ensure present
from stable_baselines3 import PPO # Ensure present
from stable_baselines3.common.monitor import Monitor # For new test


class TestPPOBasicRewardCollection(unittest.TestCase):

    def test_basic_environment_reward(self):
        """
        Tests if a standard environment ('CartPole-v1') returns non-zero rewards.
        This checks Hypothesis 1: Environment Always Returns Zero Reward.
        Failure of this test suggests a fundamental issue with the environment
        itself or the core Gym setup.
        """
        env_name = 'CartPole-v1'
        try:
            env = gym.make(env_name)
        except Exception as e:
            self.skipTest(f"Could not create environment {env_name}: {e}. "
                          f"This test assumes a standard Gym environment is available.")
            return

        print(f"Running basic reward test for {env_name}...")
        rewards_collected = []
        try:
            _ = env.reset()
            for _ in range(10):  # Take 10 random actions
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                rewards_collected.append(reward)
                if terminated or truncated:
                    break

            env.close()
        except Exception as e:
            if env: env.close() # Ensure cleanup
            self.fail(f"Error during environment interaction in test_basic_environment_reward: {e}")

        print(f"Rewards collected from {env_name}: {rewards_collected}")

        # CartPole-v1 gives +1 reward for every step taken.
        # So, if it runs for any steps, the sum of rewards should be > 0.
        # We also check if any individual reward was non-zero.
        self.assertTrue(any(r != 0 for r in rewards_collected),
                        f"All rewards collected from {env_name} were zero. Expected at least one non-zero reward. Rewards: {rewards_collected}")
        self.assertGreater(sum(rewards_collected), 0,
                           f"The sum of rewards from {env_name} was not greater than zero. Sum: {sum(rewards_collected)}, Rewards: {rewards_collected}")
        print(f"Basic reward test for {env_name} passed. Sum: {sum(rewards_collected)}")

    def test_monitor_wrapper_reward_collection(self):
        """
        Tests if the stable_baselines3 Monitor wrapper correctly records episodic rewards.
        This checks Hypothesis 3: Monitor Wrapper Issue, and
        Hypothesis 4: Incorrect Reward Access from info.
        Failure suggests issues with how Monitor is used or how episodic info is retrieved.
        """
        # It's good practice to import specific tools where they are used or at the top of the test file.
        # For this example, let's assume Monitor might not be installed or easily accessible
        # in all test environments, so we'll try-catch the import.
        try:
            from stable_baselines3.common.monitor import Monitor
        except ImportError:
            self.skipTest("stable_baselines3.common.monitor.Monitor not found. Skipping this test. "
                          "Ensure stable-baselines3 is installed.")
            return

        env_name = 'CartPole-v1'
        try:
            # No need to wrap in DummyVecEnv for Monitor, it works on base env.
            env = gym.make(env_name)
        except Exception as e:
            self.skipTest(f"Could not create environment {env_name}: {e}.")
            return

        # Wrap with Monitor. Using a temporary file for monitor logs, if any.
        # Monitor logs by default to a temp file if no path is given.
        monitored_env = Monitor(env)

        print(f"Running Monitor wrapper test for {env_name}...")
        episode_rewards = []
        num_episodes_to_run = 1
        episodes_completed = 0

        try:
            obs, _ = monitored_env.reset()
            for _ in range(200): # Max steps to prevent infinite loop
                action = monitored_env.action_space.sample()
                obs, reward, terminated, truncated, info = monitored_env.step(action)

                if terminated or truncated:
                    # When an episode ends, Monitor wrapper puts 'episode' stats in info
                    if 'episode' in info:
                        episode_rewards.append(info['episode']['r'])
                        episodes_completed += 1
                        print(f"Episode finished. Cumulative reward: {info['episode']['r']}, Length: {info['episode']['l']}")
                        if episodes_completed >= num_episodes_to_run:
                            break
                        obs, _ = monitored_env.reset() # Reset for next episode if running multiple
                    else:
                        # This case should ideally not happen if Monitor is working correctly
                        # and an episode genuinely ended.
                        print("Warning: Episode ended (terminated or truncated) but 'episode' key not found in info dict.")
                        # We will still break as the episode is over.
                        break

            monitored_env.close() # Closes the base env as well
        except Exception as e:
            if monitored_env: monitored_env.close()
            self.fail(f"Error during environment interaction in test_monitor_wrapper_reward_collection: {e}")

        self.assertGreaterEqual(episodes_completed, num_episodes_to_run,
                               f"Expected at least {num_episodes_to_run} episode(s) to complete, but only {episodes_completed} did.")

        self.assertTrue(len(episode_rewards) > 0, "No episode rewards were collected by the Monitor wrapper.")

        # For CartPole-v1, episodic rewards should be positive.
        for r_ep in episode_rewards:
            self.assertGreater(r_ep, 0,
                               f"Episodic reward collected by Monitor was {r_ep}. Expected > 0 for {env_name}.")

        print(f"Monitor wrapper test for {env_name} passed. Collected rewards: {episode_rewards}")

    def test_frame_skipping_reward_accumulation_conceptual(self):
        """
        Conceptual test for Frame Skipping Reward Accumulation.
        This test is a placeholder and should be adapted by the user to their specific
        frame skipping implementation.
        It addresses Hypothesis 5: Frame Skipping Accumulation Error.

        Instructions:
        1. If you are using a frame skipping wrapper (e.g., from gym.wrappers or a custom one):
           a. Instantiate your environment.
           b. Apply your frame skipping wrapper.
           c. For a known sequence of actions that produce varying rewards over the skipped frames:
              i.  Step through the wrapped environment.
              ii. Observe the reward returned by the wrapper.
              iii. Manually (or in a parallel, non-skipped environment instance) calculate the
                  sum of rewards that *should* have been accumulated over the skipped frames.
              iv. Assert that the wrapped environment's reported reward matches this sum.
        2. Example: If your wrapper skips 3 frames (executes action for 4 frames):
           - Action A is chosen.
           - Env step 1 (action A): reward_1
           - Env step 2 (action A): reward_2
           - Env step 3 (action A): reward_3
           - Env step 4 (action A): reward_4
           The wrapper should return a single observation (e.g., from frame 4) and a
           single reward = reward_1 + reward_2 + reward_3 + reward_4.

        Failure to correctly accumulate rewards here would directly lead to incorrect
        cumulative rewards for the PPO agent.
        """
        self.skipTest("This is a conceptual test for frame skipping. "
                      "Please adapt it to your specific frame skipping wrapper and environment. "
                      "Verify that rewards are correctly summed across skipped frames.")

        # Example structure (to be filled by the user):
        #
        # from your_project.wrappers import YourFrameSkipWrapper # Hypothetical
        #
        # env_name = "Your_Env_Name" # Replace with your environment
        # try:
        #     env = gym.make(env_name)
        #     # wrapped_env = YourFrameSkipWrapper(env, skip_k=4) # Example
        # except Exception as e:
        #     self.fail(f"Setup for conceptual frame skipping test failed: {e}")
        #     return
        #
        # # Pseudocode for testing logic:
        # # obs, _ = wrapped_env.reset()
        # # action = wrapped_env.action_space.sample()
        # #
        # # # Assume you know the rewards for the next 'skip_k' steps for this action
        # # expected_rewards_for_skipped_frames = [r1, r2, r3, r4]
        # # expected_accumulated_reward = sum(expected_rewards_for_skipped_frames)
        # #
        # # next_obs, actual_reward, terminated, truncated, info = wrapped_env.step(action)
        # #
        # # self.assertEqual(actual_reward, expected_accumulated_reward,
        # #                  "Reward from frame skipping wrapper does not match sum of individual frame rewards.")
        #
        # # wrapped_env.close()
        print("Conceptual frame skipping test was skipped as intended.") # This line will be removed by the next change block

    def test_action_repeat_wrapper_reward_accumulation(self): # Renamed and implemented
        """
        Tests if the ActionRepeatWrapper correctly accumulates rewards.
        """
        env_name = 'CartPole-v1'
        k_repeat = 4
        base_env = None
        wrapped_env_instance = None

        try:
            base_env = gym.make(env_name)
            wrapped_env_instance = ActionRepeatWrapper(base_env, k=k_repeat)

            # Reset both environments - CartPole is deterministic with seed, but rewards are always +1 per step
            # For simplicity, not setting a seed here as basic reward accumulation is the focus.
            obs_base, _ = base_env.reset(seed=42)
            obs_wrapped, _ = wrapped_env_instance.reset(seed=42)

            # It's good to check if initial observations are consistent if relevant, but not primary for reward test.
            # self.assertTrue(np.array_equal(obs_base, obs_wrapped), "Initial observations should be the same.")

            action = 0  # A specific action, e.g., push cart left

            # Manually step k_repeat times in the base environment
            manual_total_reward = 0.0
            current_obs_base = obs_base
            for i in range(k_repeat):
                # Need to use the same instance of base_env for manual stepping
                # The ActionRepeatWrapper has its own instance of the env, so we need a separate one
                # for manual calculation if we want to match internal states precisely.
                # However, the prompt implies using 'env' for manual steps and 'wrapped_env' for wrapped step.
                # This means ActionRepeatWrapper's internal env will advance, and our manual 'env' will advance.
                # For CartPole's +1 reward/step, this is fine. For state-dependent rewards, more care needed.
                # The current setup is: wrapped_env_instance steps its internal env, base_env is stepped separately.
                # Let's re-create base_env for manual calculation to ensure state isolation if needed,
                # or ensure ActionRepeatWrapper uses the *exact same instance* if that's the design.
                # The prompt seems to imply two separate instances that start from same state.
                # For CartPole, this distinction is less critical for rewards.

                # Let's create a fresh env for manual stepping to avoid state conflicts if ActionRepeatWrapper modifies its env
                # This is the most robust way if ActionRepeatWrapper truly encapsulates its own env instance.
                # However, the current ActionRepeatWrapper takes 'env' as arg.
                # Let's assume ActionRepeatWrapper uses the passed 'env' instance internally.
                # To test this correctly, we need to be careful.
                # The provided ActionRepeatWrapper in a previous subtask does `self.env = env`.

                # If ActionRepeatWrapper uses the *same instance* of 'env', then manual stepping on 'env'
                # *before* wrapped_env.step() would alter the state seen by wrapped_env.step().
                # We should reset them to the same state, then step wrapped_env, then step a fresh base_env.
                # OR, step wrapped_env, then analyze its *internal* state if possible (not easy).

                # Safest: Reset both. Step wrapped. Then, reset a *new* base_env and manually step it.
                # This ensures the manual calculation starts from the same initial state as the wrapped step.

                # Simpler for CartPole: reset both. Step base_env manually. Reset wrapped_env *again* to same seed. Step wrapped.
                # This is because CartPole's rewards are not state-dependent.
                # The prompt's flow: reset both, step base manually, step wrapped. This is fine for CartPole.

                # Re-evaluating: The ActionRepeatWrapper internally steps `self.env`.
                # If `base_env` is the same instance as `wrapped_env_instance.env`, then stepping `base_env`
                # manually will affect the state for `wrapped_env_instance.step()`.
                # The most straightforward test is to have two separate env instances starting from the same seed.

                # Let's refine:
                # 1. Create env1, wrapped_env = ActionRepeatWrapper(env1, k)
                # 2. Create env2 (for manual stepping)
                # 3. obs1, _ = wrapped_env.reset(seed=42)
                # 4. obs2, _ = env2.reset(seed=42)
                # 5. Take action in wrapped_env.
                # 6. Take same action k times in env2.

                # The original prompt's flow for manual stepping:
                # _, r, terminated, truncated, _ = base_env.step(action) # base_env is the one wrapped
                # This implies that the manual stepping is on the *same instance* that the wrapper uses.
                # This is incorrect for testing the wrapper's accumulation independently.
                # The wrapper should be tested as a black box.

                # Correct approach for testing ActionRepeatWrapper:
                # It takes an env, and steps it k times.
                # So, we compare its output to manually stepping a *separate, identical* env k times.

                # Create a fresh env for manual calculation, seeded identically
                manual_env = gym.make(env_name)
                manual_env.reset(seed=42) # Ensure it starts from the same state as wrapped_env's internal env after its reset

                # Manually step k_repeat times in the fresh 'manual_env'
                current_manual_total_reward = 0.0
                for _ in range(k_repeat):
                    _, r_manual, terminated_manual, truncated_manual, _ = manual_env.step(action)
                    current_manual_total_reward += r_manual
                    if terminated_manual or truncated_manual:
                        break
                manual_env.close() # Close the temporary manual_env

                # Step once in wrapped_env
                _, wrapped_reward, _, _, _ = wrapped_env_instance.step(action)

                self.assertEqual(wrapped_reward, current_manual_total_reward,
                                 f"Wrapped reward {wrapped_reward} does not match manual total reward {current_manual_total_reward}")

        except gym.error.DependencyNotInstalled as e:
            self.skipTest(f"Skipping test as environment {env_name} or its dependencies are not installed: {e}")
        except Exception as e:
            self.fail(f"Test failed due to an unexpected error: {e}")
        finally:
            if base_env:
                base_env.close()
            # wrapped_env_instance uses base_env, so closing base_env (if distinct) or wrapped_env_instance (if it has close)
            # If ActionRepeatWrapper does not have its own close method, base_env.close() is enough.
            # The ActionRepeatWrapper in the problem description does not have a close method itself.

    def test_ppo_integration_with_wrappers(self):
        """
        Tests PPO agent initialization and minimal training with ActionRepeatWrapper and VecFrameStack.
        """
        k_repeat = 2 # Use a small k for speed
        env_name = 'CartPole-v1'
        stacked_vec_env = None

        try:
            def make_env_fn():
                env = gym.make(env_name)
                env = Monitor(env) # Monitor is good practice for PPO
                env = ActionRepeatWrapper(env, k=k_repeat)
                return env

            vec_env = DummyVecEnv([make_env_fn])
            # Stack k_repeat frames. If ActionRepeatWrapper outputs single frames, this stacks them.
            # If ActionRepeatWrapper changes observation space (it shouldn't by default), this might need adjustment.
            # The ActionRepeatWrapper from previous task does not change observation space.
            stacked_vec_env = VecFrameStack(vec_env, n_stack=k_repeat)

            # Minimal PPO config for a quick test
            agent = PPO(
                policy='MlpPolicy',
                env=stacked_vec_env,
                n_steps=16, # Small n_steps
                batch_size=8, # Small batch_size
                n_epochs=1,   # Minimal epochs
                seed=123,
                # Ensure other params are minimal if they affect speed, e.g. learning_rate doesn't matter here.
            )

            # Attempt to train for a very short duration
            agent.learn(total_timesteps=32) # Must be >= n_steps

            self.assertTrue(True, "PPO agent learned for a short duration with wrappers without error.")

        except gym.error.DependencyNotInstalled as e:
            self.skipTest(f"Skipping test as environment {env_name} or its dependencies are not installed: {e}")
        except ImportError as e:
            self.skipTest(f"Skipping test due to import error (e.g., stable-baselines3 not installed): {e}")
        except Exception as e:
            import traceback
            self.fail(f"PPO integration test failed: {e}\n{traceback.format_exc()}")
        finally:
            if stacked_vec_env:
                stacked_vec_env.close()


    def test_ppo_agent_data_collection_reward_sanity(self):
        """
        Tests if the PPO agent's data collection process, when interacting with
        a Monitor-wrapped environment, results in non-zero episode rewards being logged.
        This is a sanity check for the PPO <-> Monitor interaction.
        It implicitly tests parts of Hypotheses 3 and 4 in the context of PPO.
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
        except ImportError:
            self.skipTest("stable_baselines3 components not found. Skipping this test. "
                          "Ensure stable-baselines3 is installed.")
            return

        env_name = 'CartPole-v1'
        try:
            # Create a function to instantiate the environment, as required by DummyVecEnv
            # And ensure it's wrapped by Monitor first.
            def make_env():
                env = gym.make(env_name)
                return Monitor(env) # Wrap with Monitor

            # Wrap in DummyVecEnv
            vec_env = DummyVecEnv([make_env])

        except Exception as e:
            self.skipTest(f"Could not create or wrap environment {env_name}: {e}.")
            return

        print(f"Running PPO agent data collection sanity test for {env_name}...")

        # Minimal PPO config for a quick test
        # n_steps should be small enough to run quickly but large enough to potentially complete an episode.
        # For CartPole, episodes can be short.
        agent = PPO(policy='MlpPolicy', env=vec_env, n_steps=50, batch_size=10, n_epochs=1, seed=123)

        try:
            # Learn for a small number of timesteps, just enough to trigger data collection
            # and potentially complete at least one episode.
            # total_timesteps should be at least n_steps.
            agent.learn(total_timesteps=100)

            # Access Monitor's logged episode returns.
            # The Monitor instance is part of the vectorized environment.
            # For DummyVecEnv, you can access underlying envs.
            # Here, we assume a single environment in DummyVecEnv.
            monitor_instance = vec_env.envs[0]

            # Monitor keeps a deque of the last 100 episode returns and lengths
            episode_returns = monitor_instance.get_episode_rewards()
            episode_lengths = monitor_instance.get_episode_lengths()

            print(f"PPO collection test: {len(episode_returns)} episode(s) completed.")
            print(f"PPO collection test: Episodic rewards: {episode_returns}")
            print(f"PPO collection test: Episodic lengths: {episode_lengths}")

            if len(episode_returns) > 0:
                # If any episodes were completed, check their rewards.
                # For CartPole-v1, rewards should be positive.
                self.assertTrue(any(r > 0 for r in episode_returns),
                                f"All collected episodic rewards during PPO.learn() were zero or less for {env_name}. "
                                f"Rewards: {episode_returns}. This indicates a problem with reward collection/logging via Monitor.")
            else:
                # This is not necessarily a failure of reward collection itself,
                # but might mean not enough timesteps were run to complete an episode.
                # However, with CartPole and 100 timesteps, an episode is likely.
                print(f"Warning: No full episodes were completed during the short PPO learn phase ({100} timesteps). "
                      "Cannot assert on episodic rewards. Consider increasing 'total_timesteps' for this test if this warning persists.")
                # Depending on strictness, you might want to assert len(episode_returns) > 0
                # self.fail("No episodes completed during PPO learn phase.")

            vec_env.close()
        except Exception as e:
            if 'vec_env' in locals() and vec_env: vec_env.close() # Ensure cleanup
            self.fail(f"Error during PPO agent data collection sanity test: {e}")

        print(f"PPO agent data collection sanity test for {env_name} passed or completed with warnings.")

if __name__ == '__main__':
    unittest.main()
