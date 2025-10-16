import numpy as np

# Define mixture components
components = {}

components[0] = [
    {
        'weight': 0.3,
        'mu_x': -3,
        'mu_y': 7,
        'sigma_x': 3.5,
        'sigma_y': 2.0,
        'rho': 0.2
    },
    {
        'weight': 0.3,
        'mu_x': 6,
        'mu_y': -8,
        'sigma_x': 2.0,
        'sigma_y': 2.5,
        'rho': -0.4
    },
    {
        'weight': 0.4,
        'mu_x': -4,
        'mu_y': 0,
        'sigma_x': 3.0,
        'sigma_y': 2.0,
        'rho': 0.3
    }
]

components[1] = [
    {
        'weight': 0.4,
        'mu_x': 0,
        'mu_y': 6,
        'sigma_x': 3.0,
        'sigma_y': 2.0,
        'rho': 0.5
    },
    {
        'weight': 0.5,
        'mu_x': 3,
        'mu_y': -4,
        'sigma_x': 2.0,
        'sigma_y': 2.0,
        'rho': -0.3
    },
    {
        'weight': 0.2,
        'mu_x': 7,
        'mu_y': 1,
        'sigma_x': 1.5,
        'sigma_y': 3.0,
        'rho': 0.2
    }
]

# 4D parameter
components[2] = [
    {
        'weight': 5,
        'mu_1': -1, 'mu_2': 5, 'mu_3': 0.3, 'mu_4': -2,
        'sigma_1': 2, 'sigma_2': 10, 'sigma_3': 0.5, 'sigma_4': 1.7,
        'corr': np.eye(4)
    },
    {
        'weight': 2,
        'mu_1': 0.05, 'mu_2': -0.1, 'mu_3': -4, 'mu_4': 0,
        'sigma_1': 0.4, 'sigma_2': 0.2, 'sigma_3': 4, 'sigma_4': 3.5
    },
    {
        'weight': 3,
        'mu_1': -3, 'mu_2': 3, 'mu_3': 0.2, 'mu_4': -1,
        'sigma_1': 2.2, 'sigma_2': 5, 'sigma_3': 1, 'sigma_4': 3.5
    }
]

# 4D parameter
components[3] = [
    {
        'weight': 3,
        'mu_1': 3, 'mu_2': -2, 'mu_3': 0.4, 'mu_4': 0.5,
        'sigma_1': 3, 'sigma_2': 2.2, 'sigma_3': 0.12, 'sigma_4': 4,
        'corr': np.eye(4)
    },
    {
        'weight': 2,
        'mu_1': -0.2, 'mu_2': 0.9, 'mu_3': 0, 'mu_4': -3.2,
        'sigma_1': 4.3, 'sigma_2': 2.1, 'sigma_3': 1, 'sigma_4': 2
    },
    {
        'weight': 4,
        'mu_1': 0.5, 'mu_2': -2, 'mu_3': -1.2, 'mu_4': -4.3,
        'sigma_1': 2, 'sigma_2': 0.2, 'sigma_3': 3.2, 'sigma_4': 1
    },
    {
        'weight': 1,
        'mu_1': 0, 'mu_2': 0, 'mu_3': -1, 'mu_4': -2.7,
        'sigma_1': 2, 'sigma_2': 0.2, 'sigma_3': 1.2, 'sigma_4': 4
    }
]