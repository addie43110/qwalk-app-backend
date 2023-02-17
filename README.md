# Quantum Walks Visualization Application Backend
Team: Addie Jordon, Jose Ossorio, Samantha Norrie, Austin Hawkins-Seagram

University of Victoria, 2023

## Description

The Quantum Walks Visualization Application is a web-application written in React as an educational tool for those interested in learning quantum walks. Due to the scarcity of available and accessible quantum walk visualizations, we have decided to create an application that models each step of a quantum walk on a line, grid, or cube (with Torus-like functionality; ie. if you walk off one side of the structure, you will appear back on the other side).

The user is able to specify the size of the structure and number of steps taken. Visualizations are then created using quantum simulators.

## Backend

Quantum walks plots are created using Qiskit and MatPlotLib in Python. The probabilities for the walks are found using the `Statevector` object which calculates each state's probabilties through evolution from instructions. You can read more here: [https://qiskit.org/documentation/stable/0.24/stubs/qiskit.quantum_info.Statevector.html](https://qiskit.org/documentation/stable/0.24/stubs/qiskit.quantum_info.Statevector.html).

The backend itself is written using Flask.


## Frontend

You can find the repository for the frontend here: [https://github.com/addie43110/qwalk-app](https://github.com/addie43110/qwalk-app).