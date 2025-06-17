import matplotlib.pyplot as plt

# display progress
def NetWorth_plot():
    plt.plot(net_worth_history)
    plt.xlabel("Episode")
    plt.ylabel("Total net worth")
    plt.title("Net worth per Episode")
    plt.grid()
    plt.show()

# plot R² history
def R_squared_plot():
    plt.plot(r2_history)
    plt.xlabel("Iteration")
    plt.ylabel("R² Score")
    plt.title("R² per Iteration")
    plt.grid()
    plt.show()


# R² and Loss Plot
def R_squared_vs_Loss():
    plt.figure(figsize=(10, 5))
    plt.plot(r2_history, label='R²')
    plt.plot([l.item() for l in losses], label='Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Metric Value")
    plt.title("R² and Loss Over Iterations")
    plt.legend()
    plt.grid()
    plt.show()