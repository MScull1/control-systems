import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

# ==========================================
# 1. Define Parameters
# ==========================================
R1 = 1000
R2 = 2000
R3 = 3000
L1 = 0.01
L2 = 0.04
C_original = 1e-9  # 10^-9
C_modified = 1e-10 # 10^-10

# The comparison factor mentioned in the text
gain_factor = 1.666e-6 

# ==========================================
# 2. Function to create the System Model
# ==========================================
def create_system(C_val):
    """
    Constructs the state-space or transfer function model.
    You must fill this in with the equations from Question 1/2.
    """
    # --- INSERT YOUR SYSTEM DEFINITION HERE ---
    # Example: If you have State Space matrices A, B, C, D derived from the circuit:
    # A = [[..., ...], [..., ...]]
    # B = [[...], [...]]
    # C_mat = [[...]] 
    # D = [[0]]
    # return ct.ss(A, B, C_mat, D)

    A = [[-(R1+R2+R3)/L1, R2/L1, 0], [R2/L2, -R2/L2, -1/L2], [0, 1/C_val, 0]]
    B = [[1/L1], [0], [0]]
    C = [[1, 0, 0]]
    D = [[0]]
    return ct.ss(A, B, C, D)
    
    # OR, if you have a Transfer Function (num/den):
    # s = ct.tf('s')
    # sys = ( ... equation using R1, R2, L1, s ... )
    
    # --- PLACEHOLDER (Delete this when you add your model) ---
    # Creating a dummy 2nd order system just so the code runs for demonstration
    # (Do not use this model for your answer!)
    s = ct.tf('s')
    sys = gain_factor / (1 + 0.001*s + 1e-7*s**2) 
    return sys

# ==========================================
# 3. Define Time and Inputs
# ==========================================
# Define simulation time vector
# Adjust the end time (0.001) based on your system's time constants
t = np.linspace(0, 0.0002, 1000) 

# Input 1: Step function (epsilon(t))
u1 = np.ones_like(t)

# Input 2: Ramp function (t * epsilon(t))
u2 = t

# Input 3: Sine function (sin(50t) * epsilon(t))
u3 = np.sin(50 * t)

# Store inputs in a list for easy looping
inputs = [
    ("Step Input", u1),
    ("Ramp Input", u2),
    ("Sine Input", u3)
]

# ==========================================
# 4. Simulation and Plotting Routine
# ==========================================
def simulate_and_plot(C_value, case_name):
    print(f"\nSimulating for {case_name} (C = {C_value})...")
    
    # Create system with the specific Capacitance
    sys = create_system(C_value)
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"System Response with C = {C_value} F", fontsize=16)

    for i, (name, u) in enumerate(inputs):
        # Calculate Forced Response
        # T is time, y is output, x is state
        T, y = ct.forced_response(sys, T=t, U=u)
        
        # Calculate Expected Behavior (Input * 1.666e-6)
        y_expected = u * gain_factor

        # Plotting
        plt.subplot(3, 1, i+1)
        plt.plot(T, y, label='System Output', linewidth=2)
        plt.plot(T, y_expected, '--r', label=f'Input * {gain_factor:.3e}')
        plt.title(f"Response to {name}")
        plt.xlabel('Time (s)')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Execute Tasks
# ==========================================

# Task A: Run with original Capacitance C = 10^-9
simulate_and_plot(C_original, "Original C")

# Task B: Run with modified Capacitance C = 10^-10
simulate_and_plot(C_modified, "Modified C")