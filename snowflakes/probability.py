from simulation import run_simulation
from particles import AgregateProbability

run_simulation(AgregateProbability(0.5), 5000, save=True)
