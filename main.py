from ttkthemes import ThemedTk
from tkinter import ttk
from system import CityGrowthSimulator

if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    style = ttk.Style()
    style.configure(".", font=("Segoe UI", 10))
    style.configure("TButton", padding=(6, 4))

    app = CityGrowthSimulator(root) # main workflow
    root.mainloop()