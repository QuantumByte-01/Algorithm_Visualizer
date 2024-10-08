import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

class AlgorithmVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Algorithm Visualizer")
        self.master.geometry("1000x800")
        self.master.configure(bg="#f0f0f0")

        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'accent': '#e74c3c',
            'text': '#2c3e50'
        }

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=self.colors['bg'])
        self.style.configure("TButton", padding=10, font=("Roboto", 12), background=self.colors['primary'], foreground="white")
        self.style.map("TButton", background=[('active', self.colors['secondary'])])
        self.style.configure("TLabel", background=self.colors['bg'], font=("Roboto", 12), foreground=self.colors['text'])
        self.style.configure("TEntry", font=("Roboto", 12))
        self.style.configure("TCombobox", padding=5, font=("Roboto", 12))

        self.main_frame = ttk.Frame(master, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(self.main_frame, text="Enhanced Algorithm Visualizer", font=("Roboto", 28, "bold"))
        label.pack(pady=20)

        instruction_label = ttk.Label(self.main_frame, text="Select an algorithm to visualize:", font=("Roboto", 16))
        instruction_label.pack(pady=10)

        self.selected_algorithm = tk.StringVar()
        algorithms = ['Bubble Sort', 'Selection Sort', 'Insertion Sort', 'Binary Search', 'Linear Search', 'Linear Regression', 'Logistic Regression']
        self.algorithm_menu = ttk.Combobox(self.main_frame, textvariable=self.selected_algorithm, values=algorithms, state="readonly", width=30)
        self.algorithm_menu.pack(pady=10)

        self.next_button = ttk.Button(self.main_frame, text="Visualize", command=self.open_visualization_window)
        self.next_button.pack(pady=20)

        self.bar_color = self.colors['primary']

    def open_visualization_window(self):
        algorithm = self.selected_algorithm.get()
        if algorithm == "":
            messagebox.showwarning("No Selection", "Please select an algorithm to visualize!")
            return

        visual_window = tk.Toplevel(self.master)
        visual_window.title(f"{algorithm} Visualization")
        visual_window.geometry("1000x800")
        visual_window.configure(bg=self.colors['bg'])

        if algorithm in ['Bubble Sort', 'Selection Sort', 'Insertion Sort']:
            self.create_sorting_visualization(visual_window, algorithm)
        elif algorithm in ['Binary Search', 'Linear Search']:
            self.create_search_visualization(visual_window, algorithm)
        elif algorithm in ['Linear Regression', 'Logistic Regression']:
            self.create_regression_visualization(visual_window, algorithm)

    def create_sorting_visualization(self, visual_window, algorithm):
        frame = ttk.Frame(visual_window, padding="20 20 20 20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, width=900, height=500, bg="white")
        self.canvas.pack(pady=20)

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.speed_scale = ttk.Scale(control_frame, from_=1, to=10, orient='horizontal', length=200)
        self.speed_scale.set(5)
        self.speed_scale.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(control_frame, text="Speed").pack(side=tk.LEFT, padx=(0, 20))

        shuffle_button = ttk.Button(control_frame, text="Shuffle", command=self.shuffle)
        shuffle_button.pack(side=tk.LEFT, padx=(0, 20))

        color_button = ttk.Button(control_frame, text="Change Color", command=self.change_bar_color)
        color_button.pack(side=tk.LEFT, padx=(0, 20))

        custom_data_button = ttk.Button(control_frame, text="Custom Data", command=self.open_custom_data_window)
        custom_data_button.pack(side=tk.LEFT, padx=(0, 20))

        visualize_button = ttk.Button(control_frame, text="Visualize", command=lambda: self.visualize_algorithm(algorithm))
        visualize_button.pack(side=tk.LEFT)

        explanation_frame = ttk.Frame(frame)
        explanation_frame.pack(fill=tk.X, pady=10)
        self.explanation_text = tk.Text(explanation_frame, height=5, wrap=tk.WORD, font=("Roboto", 12))
        self.explanation_text.pack(fill=tk.X, expand=True)
        self.explanation_text.config(state=tk.DISABLED)

        self.array = [i for i in range(1, 51)]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.array)
        self.draw_array(self.array)

    def change_bar_color(self):
        color = colorchooser.askcolor(title="Choose Bar Color")[1]
        if color:
            self.bar_color = color
            self.draw_array(self.array)

    def open_custom_data_window(self):
        custom_window = tk.Toplevel(self.master)
        custom_window.title("Enter Custom Data")
        custom_window.geometry("400x300")
        custom_window.configure(bg=self.colors['bg'])

        ttk.Label(custom_window, text="Enter comma-separated integers:").pack(pady=10)
        
        data_entry = ttk.Entry(custom_window, width=50)
        data_entry.pack(pady=10)
        
        def apply_custom_data():
            try:
                custom_data = [int(x.strip()) for x in data_entry.get().split(',')]
                if len(custom_data) < 2:
                    raise ValueError("Please enter at least 2 numbers")
                self.array = custom_data
                self.draw_array(self.array)
                custom_window.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))

        apply_button = ttk.Button(custom_window, text="Apply", command=apply_custom_data)
        apply_button.pack(pady=10)

    def draw_array(self, array, color_array=None):
        self.canvas.delete("all")
        c_width = 900
        c_height = 500
        bar_width = c_width / len(array)
        bar_spacing = 1

        max_value = max(array)
        if not color_array:
            color_array = [self.bar_color for _ in range(len(array))]

        for i, val in enumerate(array):
            scaled_height = (val / max_value) * (c_height - 20)

            x0 = i * bar_width + bar_spacing
            y0 = c_height - scaled_height
            x1 = (i + 1) * bar_width
            y1 = c_height

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color_array[i], outline="")
            self.canvas.create_text((x0 + x1) // 2, y0 - 5, anchor=tk.S, text=str(val), font=("Roboto", 8), fill=self.colors['text'])

        self.master.update_idletasks()

    def visualize_algorithm(self, algorithm):
        speed = self.speed_scale.get()
        if algorithm == 'Bubble Sort':
            self.bubble_sort(speed)
        elif algorithm == 'Selection Sort':
            self.selection_sort(speed)
        elif algorithm == 'Insertion Sort':
            self.insertion_sort(speed)

    def update_explanation(self, text):
        self.explanation_text.config(state=tk.NORMAL)
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(tk.END, text)
        self.explanation_text.config(state=tk.DISABLED)

    def bubble_sort(self, speed):
        array = self.array.copy()
        n = len(array)
        self.update_explanation("Bubble Sort repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.")
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if array[j] > array[j + 1]:
                    array[j], array[j + 1] = array[j + 1], array[j]
                    color_array = [self.bar_color for _ in range(n)]
                    color_array[j] = self.colors['accent']
                    color_array[j + 1] = self.colors['secondary']
                    self.animate_swap(array, j, j + 1, color_array, speed)
                    self.update_explanation(f"Comparing {array[j]} and {array[j+1]}. Swapped!")
                else:
                    color_array = [self.bar_color for _ in range(n)]
                    color_array[j] = self.colors['accent']
                    color_array[j + 1] = self.colors['secondary']
                    self.draw_array(array, color_array)
                    self.master.update_idletasks()
                    time.sleep(0.5 / speed)
                    self.update_explanation(f"Comparing {array[j]} and {array[j+1]}. No swap needed.")
            
            self.update_explanation(f"Pass {i+1} complete. Largest unsorted element bubbled to position {n-i-1}.")
            time.sleep(1 / speed)
        
        self.draw_array(array)
        self.update_explanation("Bubble Sort complete!")

    def selection_sort(self, speed):
        array = self.array.copy()
        n = len(array)
        self.update_explanation("Selection Sort divides the input list into two parts: a sorted portion and an unsorted portion. It repeatedly selects the smallest element from the unsorted portion and adds it to the sorted portion.")
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                color_array = [self.bar_color for _ in range(n)]
                color_array[min_idx] = self.colors['secondary']
                color_array[j] = self.colors['accent']
                self.draw_array(array, color_array)
                self.master.update_idletasks()
                time.sleep(0.5 / speed)
                
                if array[j] < array[min_idx]:
                    min_idx = j
                    self.update_explanation(f"New minimum found: {array[j]} at index {j}")
                else:
                    self.update_explanation(f"Comparing {array[j]} with current minimum {array[min_idx]}")
            
            array[i], array[min_idx] = array[min_idx], array[i]
            self.animate_swap(array, i, min_idx, color_array, speed)
            self.update_explanation(f"Swapping {array[i]} to position {i}")
            
        self.draw_array(array)
        self.update_explanation("Selection Sort complete!")

    def insertion_sort(self, speed):
        array = self.array.copy()
        n = len(array)
        self.update_explanation("Insertion Sort builds the final sorted array one item at a time. It iterates through an input array and removes one element per iteration, finds the location it belongs within the sorted list and inserts it there.")
        
        for i in range(1, n):
            key = array[i]
            j = i - 1
            self.update_explanation(f"Inserting {key} into the sorted portion of the array")
            
            while j >= 0 and key < array[j]:
                array[j + 1] = array[j]
                color_array = [self.bar_color for _ in range(n)]
                color_array[j] = self.colors['accent']
                color_array[j + 1] = self.colors['secondary']
                self.animate_swap(array, j, j + 1, color_array, speed)
                self.update_explanation(f"Shifting {array[j]} to the right")
                j -= 1
            
            array[j + 1] = key
            self.draw_array(array)
            self.master.update_idletasks()
            time.sleep(1 / speed)
            self.update_explanation(f"{key} inserted at position {j+1}")
        
        self.draw_array(array)
        self.update_explanation("Insertion Sort complete!")

    def animate_swap(self, array, i, j, color_array, speed):
        frames = 10
        initial_heights = self.get_bar_heights(array)
        for frame in range(frames + 1):
            t = frame / frames
            current_heights = self.interpolate_heights(initial_heights, array, i, j, t)
            self.draw_array_with_heights(current_heights, color_array)
            self.master.update_idletasks()
            time.sleep(0.5 / (speed * frames))

    def get_bar_heights(self, array):
        max_value = max(array)
        c_height = 500
        return [(val / max_value) * (c_height - 20) for val in array]

    def interpolate_heights(self, initial_heights, array, i, j, t):
        max_value = max(array)
        c_height = 500
        current_heights = initial_heights.copy()
        current_heights[i] = (1 - t) * initial_heights[i] + t * ((array[j] / max_value) * (c_height - 20))
        current_heights[j] = (1 - t) * initial_heights[j] + t * ((array[i] / max_value) * (c_height - 20))
        return current_heights

    def draw_array_with_heights(self, heights, color_array):
        self.canvas.delete("all")
        c_width = 900
        c_height = 500
        bar_width = c_width / len(heights)
        bar_spacing = 1

        for i, height in enumerate(heights):
            x0 = i * bar_width + bar_spacing
            y0 = c_height - height
            x1 = (i + 1) * bar_width
            y1 = c_height

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color_array[i], outline="")
            self.canvas.create_text((x0 + x1) // 2, y0 - 5, anchor=tk.S, text=str(self.array[i]), font=("Roboto", 8), fill=self.colors['text'])

        self.master.update_idletasks()

    # Search Visualizations
    def create_search_visualization(self, visual_window, algorithm):
        frame = ttk.Frame(visual_window, padding="20 20 20 20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, width=900, height=500, bg="white")
        self.canvas.pack(pady=20)

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.speed_scale = ttk.Scale(control_frame, from_=1, to=10, orient='horizontal', length=200)
        self.speed_scale.set(5) 
        self.speed_scale.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(control_frame, text="Speed").pack(side=tk.LEFT, padx=(0, 20))

        shuffle_button = ttk.Button(control_frame, text="Shuffle", command=self.shuffle_search)
        shuffle_button.pack(side=tk.LEFT, padx=(0, 20))

        color_button = ttk.Button(control_frame, text="Change Color", command=self.change_bar_color)
        color_button.pack(side=tk.LEFT, padx=(0, 20))

        custom_data_button = ttk.Button(control_frame, text="Custom Data", command=self.open_custom_data_window)
        custom_data_button.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(control_frame, text="Target:").pack(side=tk.LEFT, padx=(0, 5))
        self.target_entry = ttk.Entry(control_frame, width=10)
        self.target_entry.pack(side=tk.LEFT, padx=(0, 20))

        visualize_button = ttk.Button(control_frame, text="Search", command=lambda: self.visualize_search(algorithm))
        visualize_button.pack(side=tk.LEFT)

        explanation_frame = ttk.Frame(frame)
        explanation_frame.pack(fill=tk.X, pady=10)
        self.explanation_text = tk.Text(explanation_frame, height=5, wrap=tk.WORD, font=("Roboto", 12))
        self.explanation_text.pack(fill=tk.X, expand=True)
        self.explanation_text.config(state=tk.DISABLED)

        self.array = sorted([random.randint(1, 100) for _ in range(50)])
        self.draw_array(self.array)

    def shuffle_search(self):
        self.array = sorted([random.randint(1, 100) for _ in range(50)])
        self.draw_array(self.array)

    def visualize_search(self, algorithm):
        target = self.target_entry.get()
        if not target.isdigit():
            messagebox.showerror("Invalid Input", "Please enter a valid number to search.")
            return
        target = int(target)

        if algorithm == 'Binary Search':
            self.binary_search(target)
        elif algorithm == 'Linear Search':
            self.linear_search(target)

    def linear_search(self, target):
        speed = self.speed_scale.get()
        array = self.array.copy()
        n = len(array)
        self.update_explanation("Linear Search sequentially checks each element of the list until a match is found or the whole list has been searched.")

        for i in range(n):
            color_array = [self.bar_color for _ in range(n)]
            color_array[i] = self.colors['accent']
            self.draw_array(array, color_array)
            self.master.update_idletasks()
            time.sleep(1 / speed)

            self.update_explanation(f"Checking element at index {i}: {array[i]}")

            if array[i] == target:
                color_array[i] = self.colors['secondary']
                self.draw_array(array, color_array)
                self.update_explanation(f"Target {target} found at index {i}.")
                return

        self.update_explanation(f"Target {target} not found in the array.")

    def binary_search(self, target):
        speed = self.speed_scale.get()
        array = self.array.copy()
        low, high = 0, len(array) - 1
        self.update_explanation("Binary Search repeatedly divides the search interval in half. If the value of the search key is less than the item in the middle of the interval, narrow the interval to the lower half. Otherwise, narrow it to the upper half.")

        while low <= high:
            mid = (low + high) // 2
            color_array = [self.bar_color for _ in range(len(array))]
            for i in range(low, high + 1):
                color_array[i] = self.colors['accent']
            color_array[mid] = self.colors['secondary']
            self.draw_array(array, color_array)
            self.master.update_idletasks()
            time.sleep(1 / speed)

            self.update_explanation(f"Checking middle element at index {mid}: {array[mid]}")

            if array[mid] == target:
                self.update_explanation(f"Target {target} found at index {mid}.")
                return
            elif array[mid] < target:
                low = mid + 1
                self.update_explanation(f"Target is greater than {array[mid]}. Searching the right half.")
            else:
                high = mid - 1
                self.update_explanation(f"Target is less than {array[mid]}. Searching the left half.")

        self.update_explanation(f"Target {target} not found in the array.")

    # Regression Visualizations
    def create_regression_visualization(self, visual_window, algorithm):
        frame = ttk.Frame(visual_window, padding="20 20 20 20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=10)

        visualize_button = ttk.Button(control_frame, text="Visualize Regression", command=lambda: self.visualize_regression(algorithm))
        visualize_button.pack(side=tk.LEFT, padx=(0, 20))

        reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_regression)
        reset_button.pack(side=tk.LEFT, padx=(0, 20))

        custom_data_button = ttk.Button(control_frame, text="Custom Data", command=self.open_custom_regression_data_window)
        custom_data_button.pack(side=tk.LEFT)

        explanation_frame = ttk.Frame(frame)
        explanation_frame.pack(fill=tk.X, pady=10)
        self.explanation_text = tk.Text(explanation_frame, height=5, wrap=tk.WORD, font=("Roboto", 12))
        self.explanation_text.pack(fill=tk.X, expand=True)
        self.explanation_text.config(state=tk.DISABLED)

        self.regression_data = None
        self.reset_regression()

    def reset_regression(self):
        X = np.random.rand(100, 1) * 10
        if self.selected_algorithm.get() == 'Linear Regression':
            y = 2.5 * X + np.random.randn(100, 1)
        else:  # Logistic Regression
            y = (X > 5).astype(int).ravel()

        self.regression_data = {'X': X, 'y': y}
        self.plot_regression_data()

    def open_custom_regression_data_window(self):
        custom_window = tk.Toplevel(self.master)
        custom_window.title("Enter Custom Regression Data")
        custom_window.geometry("500x400")
        custom_window.configure(bg=self.colors['bg'])

        ttk.Label(custom_window, text="Enter comma-separated X values:").pack(pady=10)
        x_entry = ttk.Entry(custom_window, width=50)
        x_entry.pack(pady=10)

        ttk.Label(custom_window, text="Enter comma-separated y values:").pack(pady=10)
        y_entry = ttk.Entry(custom_window, width=50)
        y_entry.pack(pady=10)

        def apply_custom_data():
            try:
                X = np.array([float(x.strip()) for x in x_entry.get().split(',')]).reshape(-1, 1)
                y = np.array([float(y.strip()) for y in y_entry.get().split(',')])
                
                if len(X) != len(y):
                    raise ValueError("X and y must have the same number of values")
                if len(X) < 2:
                    raise ValueError("Please enter at least 2 data points")
                
                self.regression_data = {'X': X, 'y': y}
                self.plot_regression_data()
                custom_window.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))

        apply_button = ttk.Button(custom_window, text="Apply", command=apply_custom_data)
        apply_button.pack(pady=10)

    def plot_regression_data(self):
        self.ax.clear()
        X, y = self.regression_data['X'], self.regression_data['y']
        self.ax.scatter(X, y, color=self.colors['primary'], alpha=0.7)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('y')
        self.ax.set_title(f'{self.selected_algorithm.get()} Data')
        self.canvas.draw()

    def visualize_regression(self, algorithm):
        X, y = self.regression_data['X'], self.regression_data['y'].ravel()  # Flatten y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm == 'Linear Regression':
            model = LinearRegression()
            self.update_explanation("Linear Regression attempts to model the relationship between two variables by fitting a linear equation to observed data.")
        else:  # Logistic Regression
            model = LogisticRegression()
            self.update_explanation("Logistic Regression is used to predict the probability of a categorical dependent variable. In this case, we're using it for binary classification.")

        model.fit(X_train, y_train)

    # Ensure you plot the model correctly
        self.animate_regression(model, X_test, y_test)

    def animate_regression(self, model, X_test, y_test):
        frames = 50
        X_plot = np.linspace(min(self.regression_data['X']), max(self.regression_data['X']), 100).reshape(-1, 1)

        for frame in range(frames + 1):
            self.ax.clear()
            self.ax.scatter(self.regression_data['X'], self.regression_data['y'], color=self.colors['primary'], alpha=0.7)

            if self.selected_algorithm.get() == 'Linear Regression':
                y_plot = model.predict(X_plot)
                current_y = frame / frames * y_plot
                self.ax.plot(X_plot, current_y, color=self.colors['accent'])
                self.update_explanation(f"Fitting line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
            else:  # Logistic Regression
                y_plot = model.predict_proba(X_plot)[:, 1]
                current_y = frame / frames * y_plot
                self.ax.plot(X_plot, current_y, color=self.colors['accent'])
                self.ax.set_ylim(0, 1)
                self.update_explanation("Fitting S-shaped curve to separate two classes")

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('y')
            self.ax.set_title(f'{self.selected_algorithm.get()} Visualization')
            self.canvas.draw()
            self.master.update_idletasks()
            time.sleep(0.05)

        self.ax.scatter(X_test, y_test, color=self.colors['secondary'], marker='x', s=100, label='Test Data')
        self.ax.legend()
        self.canvas.draw()

        if self.selected_algorithm.get() == 'Linear Regression':
            mse = np.mean((model.predict(X_test) - y_test) ** 2)
            self.update_explanation(f"Final line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\nMean Squared Error on test data: {mse:.2f}")
        else:
            accuracy = model.score(X_test, y_test)
            self.update_explanation(f"Logistic Regression complete. Accuracy on test data: {accuracy:.2f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmVisualizer(root)
    root.mainloop()