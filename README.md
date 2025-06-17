# BSc Thesis – Operations Research  
**Title:** Strategic Decision-Making in Competitive Environments  
**Author:** Heider Jeffer  
**Institution:** Al Mansour University College – Dept. of Operations Research  
**Supervisor:** Dr. Hamed Al Shimrty  
**Language:** English  
**Submitted:** June 2004 | **Updated:** June 2025  

## 📘 Thesis Chapters

1. **Introduction**  
2. **Game Theory Fundamentals**  
3. **Cooperative Games – Shapley Value**  
4. **Non-Cooperative Games – Nash Equilibrium**  
5. **Case Studies and Applications**  
6. **Python Modeling**  
7. **Conclusion**  


## 💻 Python Code

### ▶ Shapley Value for Cost Sharing (Chapter 3)

```python
# Developed using Python by Heider Jeffer
import itertools

def shapley_value(players, cost_function):
    n = len(players)
    factorial = lambda x: 1 if x == 0 else x * factorial(x - 1)
    shapley = dict.fromkeys(players, 0.0)

    for player in players:
        for coalition in itertools.permutations(players):
            if player not in coalition:
                continue
            idx = coalition.index(player)
            prev_coalition = set(coalition[:idx])
            full_coalition = prev_coalition.union({player})
            marginal = cost_function(full_coalition) - cost_function(prev_coalition)
            weight = factorial(len(prev_coalition)) * factorial(n - len(prev_coalition) - 1) / factorial(n)
            shapley[player] += weight * marginal
    return shapley

# Example usage:
players = ['A', 'B', 'C']
costs = {
    frozenset(): 0,
    frozenset({'A'}): 100,
    frozenset({'B'}): 150,
    frozenset({'C'}): 130,
    frozenset({'A', 'B'}): 200,
    frozenset({'A', 'C'}): 180,
    frozenset({'B', 'C'}): 210,
    frozenset({'A', 'B', 'C'}): 270,
}

cost_function = lambda coalition: costs.get(frozenset(coalition), 0)
shapley = shapley_value(players, cost_function)

for player, val in shapley.items():
    print(f"Player {player}: {val:.2f}")
````


### ▶ Nash Equilibrium Example (Chapter 4)

```python

# Developed using Python by Heider Jeffer
import itertools

# Payoff matrix format: (Player A payoff, Player B payoff)
payoffs = {
    ('Cooperate', 'Cooperate'): (3, 3),
    ('Cooperate', 'Defect'): (0, 5),
    ('Defect', 'Cooperate'): (5, 0),
    ('Defect', 'Defect'): (1, 1)
}

strategies = ['Cooperate', 'Defect']

def find_nash_equilibria(payoffs):
    nash_equilibria = []
    for s1, s2 in itertools.product(strategies, repeat=2):
        u1, u2 = payoffs[(s1, s2)]
        alt_u1 = max(payoffs[(a, s2)][0] for a in strategies)
        alt_u2 = max(payoffs[(s1, b)][1] for b in strategies)
        if u1 == alt_u1 and u2 == alt_u2:
            nash_equilibria.append((s1, s2))
    return nash_equilibria

nash = find_nash_equilibria(payoffs)
print("Nash Equilibria:")
for eq in nash:
    print(eq)
```


## ✅ How to Run

Just save and run the code blocks above in any Python 3 environment.
No external libraries are required.
