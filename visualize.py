from train import initialize_agent, make_env
import omegaconf
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter

class FasterFFMpegWriter(animation.FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._proc.stdin.write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 

@torch.inference_mode()
def sample_actions(agent, states: torch.FloatTensor, head_weightings: torch.FloatTensor) -> torch.LongTensor:
    """
    states: (num_envs, state_shape)
    head_weightings: (num_envs, num_heads) s.t sum(head_weightings, axis=1) == [1., 1., ...]
    """
    
    "logits: (num_heads, num_envs, num_actions)"
    logits = agent.actor(states)
        
    # Combine probabilities using the weighting
    for head_index in range(agent.num_heads):
        logits[head_index] *= head_weightings[:, head_index].view(-1, 1)
    combined_logits = logits.sum(axis=0)
    
    return torch.distributions.Categorical(logits=combined_logits).sample()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./runs/PongNoFrameskip-v4__ExtrinsicCuriousPong__1__2023-03-15_16-23-02.727011", help="Folder generated by a run using train.py")
    args = parser.parse_args()
    
    # Find all models
    model_folder = f"{args.dir}/models"
    model_files = os.listdir(model_folder)
    
    # Load config
    config_file = f"{args.dir}/.hydra/config.yaml"
    config = omegaconf.OmegaConf.load(config_file)
    config.memory_size = 0
    
    # Initialize environment
    env = make_env(config.env_id, 0)()
    
    # Initialize agent
    device = torch.device("cpu")
    agent = initialize_agent(config, env.action_space, env.observation_space, device)
    agent.load(f"{model_folder}/{model_files[0]}", device=device)
    head_weightings = torch.tensor([[0, 1]])
    
    # Play
    done = False
    state = env.reset()
    frames = []
    disagreement = []
    entropy1 = []
    entropy2 = []
    while not done:
        actions = sample_actions(agent, torch.from_numpy(np.array([state])), head_weightings)
        next_state, reward, done, _ = env.step(actions.item())
        
        _, logits, action_probs = agent.actor.get_action(torch.tensor([state]))
        kl_divergence = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=logits[0]), torch.distributions.Categorical(logits=logits[1]))
        disagreement.append(kl_divergence.item())
        
        entropies = torch.distributions.Categorical(logits=logits).entropy()
        entropy1.append(entropies[0].item())
        entropy2.append(entropies[1].item())
    
        frames.append(env.render(mode="rgb_array"))
        state = next_state
        
    # Create video
    fig = plt.figure(figsize=(12,8))
    gs = plt.GridSpec(3, 4, wspace=0.2, hspace=0.2)
    img_ax = img_ax = fig.add_subplot(gs[:,0:2])
    plot_axes = [fig.add_subplot(gs[i,2:]) for i in range(gs.nrows)]
    
    # Plot first frame
    img_plot = img_ax.imshow(frames[0])
    img_ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Plot metrics
    vlines = []
    for ax, values, title in zip(plot_axes, [disagreement, entropy1, entropy2], ["KL-Divergence", "Extrinsic Entropy", "Curiosity Entropy"]):
        ax.plot(values)
        vlines.append(ax.axvline(x=0, ymin=0, ymax=max(values), color="red", linestyle="--"))
        ax.tick_params(bottom=False, left=False, labelbottom=False)
        ax.set_xlim(-1, len(frames) + 1)
        ax.set_yticks([0, max(values)])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_title(title)
    plt.show()
    
    def animate(i):
        img_plot.set_array(frames[i])
        for vline in vlines:
            vline.set_xdata([i])
        print(i)
        return [img_plot, *vlines]
        
    anim = animation.FuncAnimation(fig, animate, interval=30, frames=len(frames), blit=True)
    writer = FasterFFMpegWriter(fps=30)
    anim.save("agent.mp4", writer=writer)