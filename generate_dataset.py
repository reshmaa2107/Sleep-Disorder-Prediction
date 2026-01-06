import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 5000
np.random.seed(42)

def generate_data():
    data = []
    for _ in range(NUM_SAMPLES):
        age = np.random.randint(18, 60)
        gender = np.random.choice(['Male', 'Female'])
        occupation = np.random.choice(['Student', 'Engineer', 'Doctor', 'Artist', 'Manager'])
        daily_screen = np.round(np.random.uniform(2.0, 12.0), 1)
        night_screen = np.round(np.random.uniform(0.0, 4.0), 1)
        if night_screen > daily_screen: night_screen = daily_screen
        blue_light = np.random.choice([0, 1])
        bmi = np.round(np.random.uniform(18.5, 35.0), 1)
        heart_rate = np.random.randint(60, 95)
        stress = np.random.randint(1, 10)
        phys_act = np.random.randint(0, 120)
        snoring = np.random.choice([0, 1])
        night_walking = np.random.choice([0, 1], p=[0.95, 0.05])
        coffee = np.random.randint(0, 5)

        # Logic for Disorders
        insomnia_score = 0
        if night_screen > 2.0: insomnia_score += 3
        if blue_light == 0: insomnia_score += 1
        if stress > 6: insomnia_score += 3
        if coffee > 2: insomnia_score += 2
        
        apnea_score = 0
        if bmi > 28: apnea_score += 4
        if snoring == 1: apnea_score += 3
        if heart_rate > 80: apnea_score += 1

        if apnea_score >= 5: disorder = "Sleep Apnea"
        elif insomnia_score >= 5: disorder = "Insomnia"
        else: disorder = "Healthy"

        data.append([age, gender, occupation, daily_screen, night_screen, blue_light, bmi, heart_rate, stress, phys_act, snoring, night_walking, coffee, disorder])
    
    columns = ['Age', 'Gender', 'Occupation', 'DailyScreenTime', 'NightScreenTime', 'BlueLightFilter', 'BMI', 'HeartRate', 'StressLevel', 'PhysicalActivity', 'Snoring', 'NightWalking', 'CoffeeIntake', 'Disorder']
    pd.DataFrame(data, columns=columns).to_csv('sleep_disorder_dataset.csv', index=False)
    print("✅ Dataset Created!")

if __name__ == "__main__":
    generate_data()
