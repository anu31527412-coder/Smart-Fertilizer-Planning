import numpy as np
import pandas as pd

# Fitness function (fertilizer score)
def fertilizer_fitness(solution):
    # Example: ideal N, P, K ratio = 60:30:50
    ideal = np.array([60, 30, 50])
    return -np.sum((solution - ideal) ** 2)

# Firefly Algorithm
def firefly_algorithm(n=10, max_iter=50):
    alpha = 0.2
    beta0 = 1
    gamma = 1

    population = np.random.uniform(10, 100, (n, 3))  # N, P, K
    fitness = np.array([fertilizer_fitness(ind) for ind in population])

    for t in range(max_iter):
        for i in range(n):
            for j in range(n):
                if fitness[j] > fitness[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    population[i] += beta * (population[j] - population[i]) + alpha * (np.random.rand(3) - 0.5)
        fitness = np.array([fertilizer_fitness(ind) for ind in population])

    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx]

if __name__ == "__main__":
    best_solution, best_score = firefly_algorithm()
    print("✅ Best Fertilizer Recommendation (N, P, K):", best_solution)
    df = pd.DataFrame([best_solution], columns=["Nitrogen", "Phosphorus", "Potassium"])
    df.to_csv("../results/fertilizer_plan.csv", index=False)
    print("💾 Saved to results/fertilizer_plan.csv")
