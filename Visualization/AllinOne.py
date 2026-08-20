import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from PurePursuit import animation_data as pp_data
from Stanley import animation_data as stanley_data
from MPC import animation_data as mpc_data
from LQR import animation_data as lqr_data

class MultiVehicleAnimation:
    def __init__(self):
        self.sims = [pp_data, stanley_data, mpc_data, lqr_data]
        self.max_frames = max(len(sim["states"]) for sim in self.sims)
        self.fig, axs = plt.subplots(2, 2, figsize=(14, 14), gridspec_kw={'hspace': 0.3, 'wspace': 0.25})
        self.axs_flat = axs.flatten()
        self.ani = FuncAnimation(
            self.fig, 
            self.update_grid, 
            frames=self.max_frames, 
            interval=10, 
            repeat=False
        )
        gif_filename = "vehicle_controllers_comparison.gif"
        print(f"Exporting animation to {gif_filename}... This will take a moment.")
        writer = PillowWriter(fps=5)
        self.ani.save(gif_filename, writer=writer)
        print("GIF successfully created and saved!")
        plt.show()

    def update_grid(self, frame):
        for sim, ax in zip(self.sims, self.axs_flat):
            current_frame = min(frame, len(sim["states"]) - 1)
            ax.clear() 
            sim["plotter"].animate(
                current_frame,
                ax,
                sim["path"],
                sim["search_windows"],
                sim["closest_points"],
                sim["target_points"],
                sim["states"],
                sim["history"],
                title_name="" 
            )
            ax.set_title(sim["title"], fontsize=14, fontweight="bold", pad=20)

if __name__ == "__main__":
    MultiVehicleAnimation()
