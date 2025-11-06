
import matplotlib.pyplot as plt
def plot_price_vs_pred(df, out_path, title):
    plt.figure()
    plt.plot(df["date"], df["y_true"], label="True")
    plt.plot(df["date"], df["y_pred"], label="Pred")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Close"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
