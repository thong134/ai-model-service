import time
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.recommendation.optimizer import build_graph, optimize_route, _score_route, _measure_route

def run_benchmark(num_destinations=20, population_size=80, generations=200):
    # 1. Generate synthetic destinations
    destinations = []
    # Seed for reproducibility in benchmark
    random.seed(42)
    np.random.seed(42)
    
    for i in range(num_destinations):
        destinations.append({
            "id": f"dest_{i}",
            "name": f"Destination {i}",
            "coords": (10.0 + random.random(), 106.0 + random.random())
        })
    
    start_point = {
        "id": "start",
        "name": "Start",
        "coords": (10.5, 106.5)
    }
    
    graph = build_graph(destinations, start_point=start_point)
    start_id = "start"
    waypoints = [d["id"] for d in destinations]
    
    # 2. Benchmark Genetic Algorithm
    start_time_ga = time.time()
    solution_ga = optimize_route(
        graph, 
        start_id, 
        waypoints=waypoints,
        population_size=population_size,
        generations=generations
    )
    end_time_ga = time.time()
    ga_duration = end_time_ga - start_time_ga
    ga_distance = solution_ga.total_distance
    fitness_history = solution_ga.metadata.get("fitness_history", [])
    
    # 3. Benchmark Random Search (Baseline)
    num_random_samples = population_size * generations
    random_distances = []
    for _ in range(num_random_samples):
        route = list(waypoints)
        random.shuffle(route)
        dist, _ = _score_route(graph, start_id, route, None)
        random_distances.append(dist)
    
    avg_random_distance = np.mean(random_distances)
    best_random_distance = np.min(random_distances)
    
    # 4. Calculate Optimization Metrics
    rel_opt_avg = (avg_random_distance - ga_distance) / avg_random_distance
    rel_opt_best = (best_random_distance - ga_distance) / best_random_distance
    
    return {
        "num_destinations": num_destinations,
        "ga_time_s": ga_duration,
        "ga_distance_km": ga_distance,
        "avg_random_distance_km": avg_random_distance,
        "best_random_distance_km": best_random_distance,
        "relative_opt_vs_avg": rel_opt_avg,
        "relative_opt_vs_best": rel_opt_best,
        "fitness_history": fitness_history
    }

if __name__ == "__main__":
    print("Running Route Optimizer Benchmarks & Convergence Plot...")
    
    # Run a deep benchmark on 20 destinations for the plot
    res_plot = run_benchmark(num_destinations=25, population_size=100, generations=200)
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_plot["fitness_history"], color='blue', linewidth=2, label='Best Fitness (Distance)')
    plt.axhline(y=res_plot["avg_random_distance_km"], color='red', linestyle='--', label='Random Search Avg')
    plt.title('Genetic Algorithm Convergence (25 Destinations)')
    plt.xlabel('Generation')
    plt.ylabel('Total Distance (km)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = Path("evaluation_results/route_ga_convergence.png")
    plt.savefig(output_path)
    print(f"Convergence plot saved to {output_path}")
    
    # Normal benchmark for table
    results = []
    for n in [10, 20, 30]:
        print(f"Benchmarking with {n} destinations...")
        res = run_benchmark(num_destinations=n)
        results.append(res)
    
    # Save results to a simple text file
    with open("evaluation_results/ga_benchmark.txt", "w") as f:
        f.write("GENETIC ALGORITHM BENCHMARK RESULTS\n")
        f.write("====================================\n\n")
        for res in results:
            f.write(f"Destinations: {res['num_destinations']}\n")
            f.write(f"  GA Execution Time: {res['ga_time_s']:.4f} seconds\n")
            f.write(f"  GA Path Distance: {res['ga_distance_km']:.2f} km\n")
            f.write(f"  Random Search Avg Distance: {res['avg_random_distance_km']:.2f} km\n")
            f.write(f"  Relative Optimization (GA vs Random Avg): {res['relative_opt_vs_avg']*100:.2f}%\n")
            f.write(f"  Relative Optimization (GA vs Random Best): {res['relative_opt_vs_best']*100:.2f}%\n")
            f.write("-" * 40 + "\n")
    print("Benchmark complete. Results saved to evaluation_results/ga_benchmark.txt")
