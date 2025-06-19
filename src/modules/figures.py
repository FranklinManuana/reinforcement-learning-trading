import matplotlib.pyplot as plt
from .config import * 
import os # to help fix directory output issue when running on colab
 

# display progress
def NetWorth_plot(net_worth_history):
    plt.plot(net_worth_history)
    plt.xlabel("Episode")
    plt.ylabel("Total net worth")
    plt.title("Net worth per Episode")
    plt.grid()
    plt.savefig(os.path.join(FIGURES_DIR,"NetWorth_plot.png"))# to help fix directory output issue when running on colab
    plt.show()

# plot R² history
def R_squared_plot(r2_history):
    plt.plot(r2_history)
    plt.xlabel("Iteration")
    plt.ylabel("R² Score")
    plt.title("R² per Iteration")
    plt.grid()
    plt.savefig(os.path.join(FIGURES_DIR,"R2_plot.png"))
    plt.show()


# R² and Loss Plot
def R_squared_vs_Loss(r2_history, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(r2_history, label='R²')
    plt.plot([l.item() for l in losses], label='Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Metric Value")
    plt.title("R² and Loss Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FIGURES_DIR,"R2_loss_plot.png"))
    plt.show()
