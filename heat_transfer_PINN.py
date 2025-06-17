# Two models will be developed here to determine the temperature profiles in the solid and melt regions, respectively.
# Copyright: Prof. Cheng Luo, University of Texas at Arlington, chengluo@uta.edu
# After running, the two models are saved as Keras files in the current working directory. 
# Please refer to the README for instructions on how to upload and reuse them.

import tensorflow as tf
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

#Constants
Tth=164                # Threshold temperature for melting, °C
Tw=230                 # Wall temperature, °C
T0=25                  # Inlet temperature, °C
n=0.32                 # power-law index 
N=4                    # N=1+1/n
Pe=1                   # Péclet number
Pe1=Pe*(N+2)/N         # Péclet number for non-Newtonian fluid
TthN=(Tth-Tw)/(T0-Tw)  # Dimensionless threhold temperature
L=12.0e-3              # Total length of cylindrical and conic portions of extruder, mm
r0=1e-3                # Tube radius, mm
rf=0.875e-3            # Filament radius, mm
k=0.205                # Thermal conductivity, J/(m s °C)
p=1216                 # Mass density of polymer, kg/m**3
Cp=1863                # Heat capacity, J/(kg °C)
U=Pe*k*L/(p*Cp*rf**2)  # Feed rate, mm/s
c1=4.65                # Constant for WLF expression
c2=200.9               # Constant for WLF expression
Tr=230                 # Reference temperature for WLF expression, °C 
Mu=10400               # Reference viscosity for WLF expression, also the viscosity at Tr, Pa s**n 
Br=Mu*U**(n+1)*rf**(2*n+2)*(N+2)**(n-1)/(k*r0**(3*n+1)*(Tw-Tth))  #Dimensionless Brinkman number for non-Newtonian fluids

# Neural Network model
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer((2,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model1 = create_model()  #model 1 is used to determine temperature profiles in solid region, and the results are only valid for the area within the solid/melt interface
optimizer = tf.keras.optimizers.Adam(1e-4)

# Collocation grid
N_r = 100
N_z = 100
r_vals = np.linspace(0.01, 1.0, N_r)   #It is dimensionless coordinate with the range of 0 to 1
z_vals = np.linspace(0.0, 1.0, N_z)    #It is dimensionless coordinate with the range of 0 to 1
R, Z = np.meshgrid(r_vals, z_vals)
grid = np.stack([R.flatten(), Z.flatten()], axis=1).astype(np.float32)
X_grid_tf = tf.convert_to_tensor(grid)

# Auto-differentiation based PDE loss
@tf.function
def ad_pde_loss1(model, X):         #loss of PDE for solid region
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            u = model(X)             #u denotes dimensionless temperature, which is theta in the paper
        grads = tape1.gradient(u, X)
        u_r = grads[:, 0:1]
        u_z = grads[:, 1:2]
    u_rr = tape2.gradient(u_r, X)[:, 0:1]

    r = X[:, 0:1]
    residual = Pe*u_z -(u_rr + u_r / r)
    return tf.reduce_mean(tf.square(residual))

# Loss of boundary condition at the inlet
# Prepare boundary condition data 
r0_np_initial = np.linspace(0.01, 1.0, N_r).reshape(-1, 1)
z0_np_initial = np.zeros_like(r0_np_initial)
X0_tf_initial = tf.convert_to_tensor(np.hstack([r0_np_initial, z0_np_initial]), dtype=tf.float32)
def boundary_condition_inlet_loss(model, X0):
    return tf.reduce_mean(tf.square(model(X0) - 1))

# Loss of boundary condition at the wall
def boundary_condition_wall_loss(model):
    z_rand = tf.random.uniform((N_z, 1), 0, 1, dtype=tf.float32)
    r_bc = tf.ones_like(z_rand)
    X_bc = tf.concat([r_bc, z_rand], axis=1)
    return tf.reduce_mean(tf.square(model(X_bc)))

# Symmetry loss at r=0
@tf.function
def symmetry_loss_at_r0(model):
    r_zero = tf.zeros((N_z, 1), dtype=tf.float32)
    z_vals_tensor = tf.convert_to_tensor(z_vals.reshape(-1, 1), dtype=tf.float32)
    rz_zero = tf.concat([r_zero, z_vals_tensor], axis=1)

    with tf.GradientTape() as tape:
        tape.watch(rz_zero)
        T_r0 = model(rz_zero)

    dT_dr_r0 = tape.gradient(T_r0, rz_zero)[:, 0:1]
    return tf.reduce_mean(tf.square(dT_dr_r0))

# Training step for model 1
@tf.function
def train_step(model, X_grid_tf, X0):
    with tf.GradientTape() as tape:
        # Compute losses
        loss = ad_pde_loss1(model, X_grid_tf) + \
               10*boundary_condition_inlet_loss(model, X0) + \
               10*boundary_condition_wall_loss(model) + \
               symmetry_loss_at_r0(model)  
               # Weight for each boundary condition is 10 to ensure it is satisfied well
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop for model 1
for epoch in range(50000):
    loss_val = train_step(model1, X_grid_tf, X0_tf_initial)
    if (epoch + 1) % 10000 == 0:
        print(f"[Model 1] Epoch {epoch+1}, Loss: {loss_val.numpy():.4e}")

# Analytical solution: first 200 terms of the series solution
lambda_n = jn_zeros(0, 200)
A_n = 2 / (lambda_n * j1(lambda_n))

def analytical_solution_sum200(r, z):
    result = 0.0
    for n in range(200):
        result += A_n[n] * j0(lambda_n[n] * r) * np.exp(-lambda_n[n]**2 * z / Pe)
    return result

#  Determien solid/melt interface profile using PINN solution
r_eval = np.linspace(0.0, 1.0, 200)
z_dense = np.linspace(0.0, 1.0, 500)

z_at_TthN = []

for r_val in r_eval:
    r_grid = np.full_like(z_dense, r_val)
    input_grid = tf.convert_to_tensor(np.stack([r_grid, z_dense], axis=1), dtype=tf.float32)
    T_vals = model1(input_grid).numpy().flatten()

    # Find indices where T crosses melting temperature TthN
    diff = T_vals - TthN
    sign_change = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_change) > 0:
        i = sign_change[0]
        # Linear interpolation between z[i] and z[i+1]
        z1, z2 = z_dense[i], z_dense[i+1]
        T1, T2 = T_vals[i], T_vals[i+1]
        # Avoid division by zero if T1 == T2
        if abs(T2 - T1) > 1e-9:
            z_interp = z1 + (TthN - T1) * (z2 - z1) / (T2 - T1)
            z_at_TthN.append(z_interp)
        else:
             z_at_TthN.append(np.nan)
    else:
        z_at_TthN.append(np.nan)  # No crossing found

z_at_TthN = np.array(z_at_TthN)

# Filter out NaN values before converting to tensor
valid_indices = ~np.isnan(z_at_TthN)
r_interface_np = r_eval[valid_indices]
z_interface_np = z_at_TthN[valid_indices]

# Convert valid points to tensors for interface condition
r_interface = tf.convert_to_tensor(r_interface_np.reshape(-1, 1), dtype=tf.float32)
z_interface = tf.convert_to_tensor(z_interface_np.reshape(-1, 1), dtype=tf.float32)
X_interface = tf.concat([r_interface, z_interface], axis=1)

# Sample along z at r = 0 (or r = small)
z_vals_dense = np.linspace(0.0, 1.0, 500)
r_zero = np.zeros_like(z_vals_dense)
X_query = tf.convert_to_tensor(np.stack([r_zero, z_vals_dense], axis=1), dtype=tf.float32)

T_vals = model1(X_query).numpy().flatten()
diff = T_vals - TthN
sign_change = np.where(np.diff(np.sign(diff)))[0]

if len(sign_change) > 0:
    i = sign_change[0]
    z1, z2 = z_vals_dense[i], z_vals_dense[i+1]
    T1, T2 = T_vals[i], T_vals[i+1]
    z_r0 = z1 + (TthN - T1) * (z2 - z1) / (T2 - T1)   # z_r0 is Zs in the paper, denoting the length of the solid region
    print(f"Length of solid region is: {z_r0:.4f}")
else:
    z_r0=1.0
    print("T = TthN not reached at r = 0, and Model 2 is not applicable")

# Determien solid/melt interface profile using analytical solution
r_vals_analytic = np.linspace(0.0, 1.0, 200)
z_interface_analytic = []

# Loop over r and solve T(r, z) = TthN for z
for r in r_vals_analytic:
    def f(z):  # Root-finding function: T(r,z) - TthN = 0
        return analytical_solution_sum100(r, z) - TthN

    try:
        sol = root_scalar(f, bracket=[0, 1], method='brentq')
        z_interface_analytic.append(sol.root)
    except ValueError:
        z_interface_analytic.append(np.nan)

z_interface_analytic = np.array(z_interface_analytic)

#Plot solid/melt interface profiles at three cross-sections determined by PINN and analytical solution
r_test = np.linspace(0.0, 1.0, 200)
for z_fixed in [0.0, 0.1, 0.2]:
    z_grid = np.full_like(r_test, z_fixed)
    X_plot = tf.convert_to_tensor(np.stack([r_test, z_grid], axis=1), dtype=tf.float32)
    T_pred = model1(X_plot).numpy().flatten()
    T_true = analytical_solution_sum200(r_test, z_fixed)
    plt.plot(r_test, T_true*(T0-Tw)+Tw, '-',label=f'Analytical z={z_fixed:.1f}')
    plt.plot(r_test, T_pred*(T0-Tw)+Tw, '--',label=f'Predicted z={z_fixed:.1f}')
    plt.legend()

plt.xlim(0, 1)
plt.ylim(25, 230)  # optional, to fix y-axis range
yticks = np.arange(-25, 231, 50)

# add tick at 230 but don't add label for it
new_yticks = np.append(yticks, 230)
plt.yticks(new_yticks)

# Now hide the label for 230 by setting it to an empty string
labels = [str(tick) if tick != 230 else '' for tick in new_yticks]
plt.yticks(new_yticks, labels)
# Add a custom label for 230 slightly above its position
plt.text(-0.04, 230, '230', va='bottom', ha='center', fontsize=10)
plt.xlabel('R')
plt.ylabel('T(°C)')
plt.tight_layout()
plt.show()

#Plot the solid/melt interface profiles determined by Model 1 and analytical solution, respectively
plt.figure(figsize=(6,4))
plt.plot(r_interface_np, z_interface_np, '--',label=f'Predicted interface')
plt.plot(r_vals_analytic, z_interface_analytic, '-',label=f'Analytical interface')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('R')
plt.ylabel(r'$Z_i$')
plt.tight_layout()
plt.show()

#Fixed points to check
check_points = [(0.0, 0.1), (0.0, 0.2)]

print("\nTemperature Comparison at (R,Z):")
print(" R     Z     Predicted (°C)    Analytical (°C)")
print("------------------------------------------------")

for R_val, Z_val in check_points:
    input_tensor = tf.convert_to_tensor([[R_val, Z_val]], dtype=tf.float32)
    T_pred_norm = model1(input_tensor).numpy().flatten()[0]
    T_pred = T_pred_norm * (T0 - Tw) + Tw  # Dimensional predicted temperature

    T_ana_norm = analytical_solution_sum200(R_val, Z_val)
    T_ana = T_ana_norm * (T0 - Tw) + Tw    # Dimensional analytical temperature

    print(f"{R_val:>3.1f}   {Z_val:>3.2f}     {T_pred:>10.2f}        {T_ana:>10.2f}")

# Model 2 is used to determine temperatue profiles in the melt region, with the same NN architecture as Model 1. Its results are only valid outside the solid/melt interface

model2= create_model()   

optimizer2 = tf.keras.optimizers.Adam(1e-4)

# Loss of PDE for the melt region
def ad_pde_loss2(model, X):
    with tf.GradientTape() as tape22:
        tape22.watch(X)
        with tf.GradientTape() as tape21:
            tape21.watch(X)
            u = model(X)           #  #u denotes dimensionless temperature, which is phi in the paper
        grads = tape21.gradient(u, X)
        u_r = grads[:, 0:1]
        u_z = grads[:, 1:2]
    u_rr = tape22.gradient(u_r, X)[:, 0:1]

    r = X[:, 0:1]
    Alpha=10**((-c1*(Tw+u*(Tth-Tw)-Tr))/(c2+(Tw+u*(Tth-Tw)-Tr)))
    residual = Pe1*(1 - r**N) * u_z - (u_rr + u_r / r) +Br*Alpha*(N+2)**2*r**N
    return tf.reduce_mean(tf.square(residual)), tf.reduce_mean(tf.abs((u_rr + u_r / r))), tf.reduce_mean(tf.abs(Br*Alpha*(Tth-Tw)*(N+2)**2*r**N))

# Loss of interface condition for the melt region, with the interface profile calculated using Model 1
def interface_condition_loss(model):
    T_pred = model(X_interface)
    return tf.reduce_mean(tf.square(T_pred - 1.0))

# Training step for model 2 
@tf.function
def train_step_model2(model, optimizer_m2, X_grid_tf):
    with tf.GradientTape() as tape:
        loss = ad_pde_loss2(model, X_grid_tf)[0] + \
               10*boundary_condition_wall_loss(model) + \
               symmetry_loss_at_r0(model) + \
               50*interface_condition_loss(model) 
	      # Weights for boundary and interface conditions are 10 and 50, respectively, to ensure they are satisfied well
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer_m2.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop for model2
for epoch in range(50000):
    # Pass both the model2 and its optimizer to the training step
    loss_val = train_step_model2(model2, optimizer2, X_grid_tf)
    if (epoch + 1) % 10000 == 0:
        print(f"[Model 2] Epoch {epoch+1}, Loss: {loss_val.numpy():.4e}")

# Plot temperature profiles at four cross-sections in the melt region
r_test = np.linspace(0.0, 1.0, 100)
for z_fixed in [0.4,0.6, 0.8, 1.0]:
    z_grid = np.full_like(r_test, z_fixed)
    X_plot = tf.convert_to_tensor(np.stack([r_test, z_grid], axis=1), dtype=tf.float32)
    T_pred = model2(X_plot).numpy().flatten()
    plt.plot(r_test, T_pred*(Tth-Tw)+Tw, '--',label=f'Predicted z={z_fixed:.1f}') #Dimensional temperature
    plt.legend()
plt.xlim(0, 1)
plt.ylim(160, 230)  # optional, to fix y-axis range
yticks = np.arange(160, 230, 10)
plt.xlabel('R')
plt.ylabel('T(°C)')
plt.tight_layout()
plt.show()

# Fixed points to evaluate
points_to_check = [
    (0.0, 0.4),
    (0.0, 0.6),
    (0.0, 0.8),
    (0.0, 1.0),
]

# Print header
print("\nTemperature Comparison at Specific (R,Z) Points:")
print(" R     Z     Model 2 (°C)")
print("------------------------------------------------------------")

# Loop through and compute
for R_val, Z_val in points_to_check:
    input_point = tf.convert_to_tensor([[R_val, Z_val]], dtype=tf.float32)
    T_pred_norm= model2(input_point).numpy().flatten()[0]
    T_pred = T_pred_norm * (Tth - Tw) + Tw
    print(f"{R_val:>3.1f}   {Z_val:>3.2f}   {T_pred:>10.2f}")

# Save Model 1
model1.save("model1.keras")
print("Model 1 saved  as 'model1.keras' in the current working directory.")

# Save Model 2
model2.save("model2.keras")
print("Model 2 saved as 'model2.keras' in the current working directory.")
