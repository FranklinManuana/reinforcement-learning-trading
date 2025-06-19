import matplotlib.pyplot as plt
 

# display progress
def NetWorth_plot(net_worth_history):
    plt.plot(net_worth_history)
    plt.xlabel("Episode")
    plt.ylabel("Total net worth")
    plt.title("Net worth per Episode")
    plt.grid()
    plt.show()
    plt.savefig("output/figures/NetWorth_plot.png")

# plot R² history
def R_squared_plot(r2_history):
    plt.plot(r2_history)
    plt.xlabel("Iteration")
    plt.ylabel("R² Score")
    plt.title("R² per Iteration")
    plt.grid()
    plt.show()
    plt.savefig("output/figures/R2_plot.png")


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
    plt.show()
    plt.savefig("output/figures/R2_loss_plot.png")
