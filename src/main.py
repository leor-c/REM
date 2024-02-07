from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from loguru import logger

from agent import Agent, AgentLS
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import ActorCritic, ActorCriticLS
from models.world_model import RetNetWorldModel, POPRetNetWorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed, GradNormInfo
from dataset import get_dataloader


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        project_root = Path(hydra.utils.get_original_cwd())
        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / f'{config_name}.yaml'
            logger.debug(f"abs path: {config_path.absolute()}")
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            if not cfg.common.metrics_only_mode:
                shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
                shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        disable_saving = cfg.common.metrics_only_mode
        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save, disable_saving=disable_saving)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save, disable_saving=disable_saving)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save, disable_saving=disable_saving)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        vgg_lpips_rel_path = cfg.tokenizer.vgg_lpips_ckpt_path
        cfg.tokenizer.vgg_lpips_ckpt_path = (project_root / vgg_lpips_rel_path).absolute()
        tokenizer = instantiate(cfg.tokenizer)

        if cfg.world_model.compute_states_parallel:
            wm_class = POPRetNetWorldModel
        else:
            wm_class = RetNetWorldModel

        world_model = wm_class(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model.retnet), **cfg.world_model)
        if cfg.actor_critic.name == "LSAC":
            actor_critic = ActorCriticLS(**cfg.actor_critic, act_vocab_size=env.num_actions, token_embed_dim=cfg.tokenizer.embed_dim, tokens_per_obs=cfg.world_model.retnet.tokens_per_block-1, context_len=self.cfg.world_model.context_length)
            self.agent = AgentLS(tokenizer, world_model, actor_critic).to(self.device)
        else:
            assert cfg.actor_critic.name == "AC"
            actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
            self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)

        logger.info(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        logger.info(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        logger.info(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.AdamW(self.agent.tokenizer.parameters(), lr=cfg.training.tokenizer.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.world_model.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.AdamW(self.agent.actor_critic.parameters(), lr=cfg.training.actor_critic.learning_rate)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            logger.info(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0) and (epoch >= self.cfg.training.tokenizer.start_after_epochs):
                self.test_dataset.clear()
                if epoch == self.cfg.common.epochs:
                    self.cfg.collection.test.config.num_episodes = 100
                to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                to_log += self.eval_agent(epoch)

            if self.cfg.training.should and epoch % 50 == 0 and not self.cfg.common.metrics_only_mode:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time)})
            for metrics in to_log:
                logger.info(f"Epoch {epoch}: {metrics}")
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(
                epoch,
                self.agent.tokenizer,
                self.optimizer_tokenizer,
                sequence_length=1,
                sample_from_start=True,
                context_len=0,
                **cfg_tokenizer
            )
        self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component(
                epoch,
                self.agent.world_model,
                self.optimizer_world_model,
                sequence_length=self.cfg.common.sequence_length,
                sample_from_start=True,
                tokenizer=self.agent.tokenizer,
                context_len=self.cfg.world_model.context_length,
                **cfg_world_model
            )
        self.agent.world_model.eval()

        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(
                epoch, 
                self.agent.actor_critic, 
                self.optimizer_actor_critic,
                sequence_length=1 + self.cfg.training.actor_critic.burn_in, 
                sample_from_start=False,
                tokenizer=self.agent.tokenizer, 
                world_model=self.agent.world_model, 
                context_len=self.cfg.world_model.context_length,
                **cfg_actor_critic
            )
        self.agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, epoch: int, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                        batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float],
                        sequence_length: int, sample_from_start: bool,
                        context_len: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        grad_norms_info = GradNormInfo()

        dataloader = get_dataloader(
            self.train_dataset,
            context_len,
            sequence_length,
            batch_num_samples,
            shuffle=True,
            padding_strategy='right' if sample_from_start else 'left'
        )

        data_iter = iter(dataloader)
        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                assert (batch['mask_padding'].sum(dim=1) > context_len).all()
                batch = self._to_device(batch)

                losses, info = component.compute_loss(batch, **kwargs_loss)

                losses = losses / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            grad_norm = np.sqrt(sum([p.grad.norm(2).item()**2 for p in component.parameters() if p.grad is not None]))
            grad_norms_info(grad_norm)

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        epoch_info = {}
        for k, v in grad_norms_info.get_info().items():
            epoch_info[f"{str(component)}/train/{k}"] = v
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses, **epoch_info}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1, context_length=0)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(
                self.agent.world_model, 
                cfg_world_model.batch_num_samples, 
                sequence_length=self.cfg.common.sequence_length, 
                context_length=self.cfg.world_model.context_length, 
                tokenizer=self.agent.tokenizer
            )

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions and not self.cfg.common.metrics_only_mode:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, context_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        dataloader = get_dataloader(
                self.test_dataset,
                context_length,
                sequence_length,
                batch_num_samples,
                shuffle=True,
                padding_strategy='right'
            )
        num_batches = int(np.ceil(len(dataloader) / sequence_length)) if len(dataloader) > sequence_length else len(dataloader)
        data_iter = iter(dataloader)
        for batch_i in range(num_batches):
            batch = next(data_iter)
            assert (batch['mask_padding'].sum(dim=1) > context_length).all()
            batch = self._to_device(batch)

            losses, info = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        if isinstance(self.agent.actor_critic, ActorCriticLS):
            outputs.observations = torch.clamp(self.agent.tokenizer.decode(outputs.observations, should_postprocess=True), 0, 1).mul(255).byte()

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            if not self.cfg.common.metrics_only_mode:
                self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        logger.info(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()


config_name = "config"
@hydra.main(config_path="../config", config_name=config_name, version_base='1.1')
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()

