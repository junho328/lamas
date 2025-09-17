import os
import numpy as np
from tqdm import tqdm
import torch
from marft.mas import MAS
from marft.utils.logger import Logger

class CodingRunner:
    """Runner class to perform training, evaluation. and data collection. See parent class for details."""

    def __init__(self, config):
        self.num_agents = config["num_agents"]
        self.all_args = config["all_args"]
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.algo = self.all_args.algorithm_name
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]

        self.mas = MAS(
            model_path=self.all_args.model_name_or_path, 
            context_window=self.all_args.context_window,
            max_new_tokens=self.all_args.max_new_tokens, 
            num_agents=self.num_agents,
            profile_path=self.all_args.profile_path,
            algo=self.algo,
            normalization_mode=self.all_args.normalization_mode,
            load_path=self.all_args.load_path,
        )

        if self.algo == "APPO":
            from marft.algorithms import APPOTrainer
            from marft.buffers.action_level_buffer import ActionBuffer
            self.trainer = APPOTrainer(self.all_args, self.mas)
            self.buffer = ActionBuffer(self.all_args, self.num_agents)
        elif self.algo == "TPPO":
            from marft.algorithms import TPPOTrainer
            from marft.buffers.token_level_buffer import TokenBuffer
            self.trainer = TPPOTrainer(self.all_args, self.mas)
            self.buffer = TokenBuffer(self.all_args, self.num_agents, self.mas.tokenizer.pad_token_id)
        else:
            raise NotImplementedError
        
        self.run_dir = config["run_dir"]
        self._make_log_dir()
        
        # Initialize logger with wandb support
        wandb_config = {
            "algorithm": self.algo,
            "num_agents": self.num_agents,
            "model_name": self.all_args.model_name_or_path,
            "dataset_name": self.all_args.dataset_name,
            "learning_rate": self.all_args.lr,
            "critic_lr": self.all_args.critic_lr,
            "ppo_epochs": self.all_args.ppo_epoch,
            "num_mini_batch": self.all_args.num_mini_batch,
            "episode_length": self.all_args.episode_length,
            "n_rollout_threads": self.all_args.n_rollout_threads,
            "context_window": self.all_args.context_window,
            "max_new_tokens": self.all_args.max_new_tokens,
            "seed": self.all_args.seed,
        }
        
        self.logger = Logger(
            log_dir=self.log_dir,
            use_wandb=getattr(self.all_args, 'use_wandb', False),
            wandb_project=getattr(self.all_args, 'wandb_project', 'marft-coding'),
            wandb_entity=getattr(self.all_args, 'wandb_entity', None),
            wandb_run_name=getattr(self.all_args, 'wandb_run_name', None),
            config=wandb_config
        )


    def run(self):
        training_steps = 0
        next_obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = next_obs.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        progress_bar = tqdm(total=episodes, desc=f"Start running...", position=0, leave=True)

        for episode in range(episodes):

            # if eval
            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                torch.cuda.empty_cache()
                # self.eval(training_steps)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            for step in range(self.episode_length):
                torch.cuda.empty_cache()
                rollout_obs, actions, action_tokens, values, log_probs = self.mas.infer_for_rollout(self.buffer.obs[self.buffer.cur_batch_index, step])
                next_obs, rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)

                for i in range(self.n_rollout_threads):
                    global_step = episode * self.episode_length * self.n_rollout_threads + step * self.n_rollout_threads + i
                    if dones[i, 0]:
                        episodic_return = infos[i]['episodic_return']
                        self.logger.add_scalar("episodic return", episodic_return, global_step)

            self.before_update()
            train_infos = self.trainer.train(self.buffer, total_num_steps)
            training_steps += 1

            self.buffer.after_update()

            # post process
            # save model
            if (episode == episodes - 1) or ((episode + 1) % self.all_args.save_interval == 0):
                self.save(training_steps)

            # log info
            if episode % self.log_interval == 0:
                avg_step_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1])
                progress_bar.set_description(
                    f"Episode {episode}/{episodes}"
                    f"(total step num: {total_num_steps} | average step reward: {avg_step_reward})",
                )
                train_infos["avg_step_rewards"] = avg_step_reward
                self.log_train(train_infos, total_num_steps)
                self.logger.add_scalar('average_reward', avg_step_reward, training_steps)
            progress_bar.update(1)

    def insert(self, data):
        next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs = data
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        self.buffer.insert(next_obs, actions, rollout_obs, values, rewards, masks, action_tokens, log_probs)

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        values = self.mas.get_next_values(self.buffer.obs[self.buffer.cur_batch_index, -1])
        self.buffer.compute_gae_and_returns(values)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.logger.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, training_steps):
        print(f"start evaluating......")
        eval_obs = self.eval_envs.reset()
        eval_env_infos = {}
        _, eval_actions, _, _, _ = self.mas.infer_for_rollout(eval_obs, evaluating=True)
        eval_next_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
        effective_eval_rewards = []
        for i in range(self.num_agents):
            eval_env_infos[f"eval_rewards/{self.mas.profiles[i]['role']}"] = eval_rewards[:, i]
            if self.mas.profiles[i]['with_answer']:
                effective_eval_rewards.extend(eval_rewards[:, i])
        eval_env_infos["eval_rewards/effective"] = effective_eval_rewards
        print(f"eval rewards: {np.mean(effective_eval_rewards)}")
        self.log_eval(eval_env_infos, training_steps)

        # eval_dones_env = np.all(eval_dones, axis=1)

        # for eval_i in range(self.n_eval_rollout_threads):
        #     if eval_dones_env[eval_i]:
        #         eval_episode += 1
        #         eval_episode_rewards.append(eval_rewards[eval_i])

        # if eval_episode >= self.all_args.eval_episodes:
        #     eval_episode_rewards = np.array(eval_episode_rewards)
        #     eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
        #     print("total_num_steps: ", total_num_steps)
        #     print("eval reward is {}.".format(np.mean(eval_episode_rewards)))
        #     self.log_eval(eval_env_infos, total_num_steps)
        #     break

    def _make_log_dir(self):
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / "checkpoints/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def log_eval(self, eval_infos, training_steps):
        for k, v in eval_infos.items():
            if len(v) > 0:
                self.logger.add_scalars(k, {k: np.mean(v)}, training_steps)

    def save(self, steps):
        """Save the MAS policies and critic networks."""
        self.mas.save(self.save_dir, steps)
        self.trainer.save_optimizers(self.save_dir, steps)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.mas.restore(model_dir)
