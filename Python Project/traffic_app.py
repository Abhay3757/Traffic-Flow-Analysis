import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_data(file_path=None):
    if file_path:
        data = pd.read_csv(file_path)
    else:
        data = generate_simulated_data()
    return data

def generate_simulated_data():
    np.random.seed(0)
    data = {
        'hour': list(range(24)),
        'traffic_volume': np.random.randint(100, 500, 24),
        'road_condition': [random.choice(['Clear', 'Accident', 'Roadblock', 'Construction']) for _ in range(24)],
        'weather': [random.choice(['Clear', 'Rain', 'Fog']) for _ in range(24)]
    }
    return pd.DataFrame(data)

def preprocess_data(data):
    data['traffic_volume'] = data['traffic_volume'].fillna(data['traffic_volume'].mean())
    return data

def train_model(data):
    X = data[['hour']]
    y = data['traffic_volume']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_traffic(model, hour, condition, weather):
    traffic_factor = 1.0
    if condition == 'Accident':
        traffic_factor += 0.5
    elif condition == 'Roadblock':
        traffic_factor += 0.3
    if weather == 'Rain':
        traffic_factor += 0.2
    predicted_volume = model.predict([[hour]])[0] * traffic_factor
    return max(0, int(predicted_volume))

class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Traffic Flow Analysis")
        self.root.geometry("700x800")
        self.root.configure(bg="#2C2C2E")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#2C2C2E", foreground="#E1E1E1", font=("Helvetica", 12))
        style.configure("TButton", background="#007AFF", foreground="#FFFFFF", font=("Helvetica", 12, "bold"))
        style.configure("TCombobox", background="#FFFFFF", foreground="#000000", font=("Helvetica", 12))
        
        # Additional style for frame
        style.configure("TFrame", background="#1C1C1E")

        self.data = preprocess_data(generate_simulated_data())
        self.model = train_model(self.data)

        title_label = ttk.Label(self.root, text="Traffic Flow Analysis", font=("Helvetica", 24, "bold"), foreground="#FFD700", anchor="center")
        title_label.pack(pady=15)

        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(padx=25, pady=20, fill='x')

        region_label = ttk.Label(main_frame, text="Select Region:")
        region_label.grid(row=0, column=0, sticky="W", pady=5)
        self.region_var = tk.StringVar(value='University Campus')
        region_menu = ttk.Combobox(main_frame, textvariable=self.region_var, values=[
            'University Campus', 'Tech Park Road', 'Hospital Road', 'Architecture Block', 'Dental College'
        ], state="readonly", style="TCombobox")
        region_menu.grid(row=0, column=1, padx=10, pady=5)

        condition_label = ttk.Label(main_frame, text="Road Condition:")
        condition_label.grid(row=1, column=0, sticky="W", pady=5)
        self.condition_var = tk.StringVar(value='Clear')
        condition_menu = ttk.Combobox(main_frame, textvariable=self.condition_var, values=['Clear', 'Accident', 'Roadblock', 'Construction'], state="readonly", style="TCombobox")
        condition_menu.grid(row=1, column=1, padx=10, pady=5)

        weather_label = ttk.Label(main_frame, text="Weather:")
        weather_label.grid(row=2, column=0, sticky="W", pady=5)
        self.weather_var = tk.StringVar(value='Clear')
        weather_menu = ttk.Combobox(main_frame, textvariable=self.weather_var, values=['Clear', 'Rain', 'Fog'], state="readonly", style="TCombobox")
        weather_menu.grid(row=2, column=1, padx=10, pady=5)

        predict_button = ttk.Button(main_frame, text="Predict Traffic", command=self.predict_traffic, style="TButton")
        predict_button.grid(row=3, column=0, columnspan=2, pady=20)

        self.result_label = ttk.Label(self.root, text="Predicted Traffic Volume: -", font=('Helvetica', 14, 'bold'), anchor="center", foreground="#FFD700")
        self.result_label.pack(pady=10)

        self.suggestion_label = ttk.Label(self.root, text="", font=('Helvetica', 12), anchor="center", foreground="#E1E1E1")
        self.suggestion_label.pack(pady=10)

        graph_button = ttk.Button(self.root, text="Show Traffic Graph", command=self.show_graph, style="TButton")
        graph_button.pack(pady=10)

        map_frame = ttk.Frame(self.root)
        map_frame.pack(pady=20)

        map_image = Image.open(r"C:\Users\abhaa\Python Project\map_image.png")
        map_image = map_image.resize((600, 400), Image.LANCZOS)
        self.map_photo = ImageTk.PhotoImage(map_image)

        map_label = ttk.Label(map_frame, image=self.map_photo, relief="solid", borderwidth=2)
        map_label.pack(padx=10, pady=10)

        # Style for main frame
        self.style_frame(main_frame)

    def style_frame(self, frame):
        # Removed background setting for ttk widgets
        for child in frame.winfo_children():
            if isinstance(child, tk.Frame):  # Only set background for tk.Frame widgets
                child.configure(bg="#1C1C1E")

    def predict_traffic(self):
        current_hour = pd.Timestamp.now().hour
        condition = self.condition_var.get()
        weather = self.weather_var.get()
        prediction = predict_traffic(self.model, current_hour, condition, weather)
        self.result_label.config(text=f"Predicted Traffic Volume: {prediction:,}")
        
        suggestion = self.get_route_suggestion(prediction)
        self.suggestion_label.config(text=suggestion)

    def get_route_suggestion(self, prediction):
        if prediction > 400:
            return "Route Suggestion: Take an alternative route; heavy congestion expected."
        elif 300 < prediction <= 400:
            return "Route Suggestion: Moderate traffic; proceed with caution."
        else:
            return "Route Suggestion: Clear route ahead."

    def show_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['hour'], self.data['traffic_volume'], label='Traffic Volume', color='#FF7F50', marker='o')
        plt.fill_between(self.data['hour'], self.data['traffic_volume'], color='lightblue', alpha=0.5)
        plt.xlabel('Hour of Day')
        plt.ylabel('Traffic Volume')
        plt.title('Traffic Volume Throughout the Day', fontsize=16)
        plt.xticks(self.data['hour'])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()






































