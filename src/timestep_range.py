
class TimestepRange:
    def __init__(self, train_timesteps, schedules=[5, 10, 15], device="cpu"):
        # split into quantiles (easiest = largest logSNR)
        # qs = torch.linspace(0, max_timestep, n_stages+1).to(device)
        
        self.train_timesteps = train_timesteps
        self.schedule = schedules
        self.stage = 0
        self.timestep = self.train_timesteps[0]
        
    def advance(self, epoch, verbose=False):
        if epoch in self.schedule and self.stage+1 < len(self.train_timesteps):
            self.stage += 1
            self.timestep = self.train_timesteps[self.stage]
            print(f"Advanced to stage {self.stage} with timestep {self.timestep} at epoch {epoch}") if verbose else None
