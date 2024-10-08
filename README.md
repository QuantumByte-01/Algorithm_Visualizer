# Algorithm Visualizer

Algorithm Visualizer is an interactive Python application that helps users understand and visualize various algorithms, including sorting algorithms, search algorithms, and regression models. Built with Tkinter, Matplotlib, and scikit-learn, this tool provides an engaging way to learn about fundamental computer science and machine learning concepts.

## Features

- Visualize multiple algorithms:
  - Sorting: Bubble Sort, Selection Sort, Insertion Sort
  - Searching: Linear Search, Binary Search
  - Regression: Linear Regression, Logistic Regression
- Interactive GUI with customizable options
- Step-by-step visualization with explanations
- Ability to input custom data
- Adjustable visualization speed
- Color customization for visual elements

## Requirements

To run the Algorithm Visualizer, you need:

- Python 3.x
- Tkinter (usually comes pre-installed with Python)
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using pip:

```
pip install numpy matplotlib scikit-learn
```

## How to Run

1. Ensure you have all the required packages installed.
2. Save the provided code in a file named `algorithm_visualizer.py`.
3. Run the script using Python:

```
python3 algorithm_visualizer.py
```

## Usage

1. When you run the application, you'll see the main window with a dropdown menu to select an algorithm.
2. Choose an algorithm from the dropdown menu and click "Visualize" to open the visualization window.
3. In the visualization window, you can:
   - Adjust the visualization speed using the slider
   - Shuffle the data (for sorting and searching algorithms)
   - Change the color of the visual elements
   - Input custom data
   - Start the visualization
4. For search algorithms, enter a target value to search for.
5. For regression algorithms, you can visualize the fitting process and see the results on test data.

## Customization

- You can modify the color scheme by changing the values in the `colors` dictionary in the `__init__` method of the `AlgorithmVisualizer` class.
- To add new algorithms, extend the `visualize_algorithm` method and create corresponding visualization methods.

