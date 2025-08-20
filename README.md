# 🎓 Campus Allocation System

## 📌 Overview
The **Campus Allocation System** is a Python-based solution for assigning students to classes under strict scheduling and capacity constraints.  
It ensures that:
- Each class respects **minimum and maximum enrollment sizes**.  
- Students are matched to classes based on **time preferences**.  
- At least a specified **minimum satisfaction** level is achieved (number of students assigned to one of their top-5 preferred time slots).  

This project applies **network flow and optimization techniques** to solve real-world scheduling challenges faced in universities.

---

## ⚡ Features
- ✅ Assigns each student to **exactly one class**.  
- ✅ Ensures all classes respect **capacity bounds (min/max)**.  
- ✅ Maximizes the number of students placed into one of their **top-5 time preferences**.  
- ✅ Returns `None` if **no feasible allocation** exists.  
- ✅ Includes **built-in test cases** for validation.

  ## 🧮 Algorithmic Approach

The allocation problem is modeled as a **Min-Cost Max-Flow (MCMF) problem with lower/upper bound constraints**, built upon the **Ford-Fulkerson framework** of finding augmenting paths.

### Graph Construction
- **Source → Students**: Each student is connected from the source with capacity `1`.  
- **Students → Classes**: Each student connects to eligible classes.  
  - Edge cost = `-1` if the class time is within their **top-5 preferences** (to encourage satisfaction).  
  - Otherwise, cost = `0`.  
- **Classes → Sink**: Each class connects to the sink with capacity `(maxCap - minCap)`.  
  - **Demands** enforce **minimum class sizes**.  

### Feasibility Check
- A **circulation step** ensures that **lower bound capacities** can be satisfied before optimization begins.  

### Optimization
- The system applies a **successive shortest path algorithm with potentials** (a refinement of Ford-Fulkerson for min-cost flows).  
- Each augmentation is equivalent to finding a shortest augmenting path in the residual network.  
- A flow is feasible **only if**:
  - All students are allocated, **and**  
  - Total satisfaction ≥ required minimum.  

✅ This ensures that the allocation is both **valid (meets constraints)** and **optimal (maximizes satisfaction)**.  

---

## ⏱️ Complexity Analysis

Let:  
- **n** = number of students  
- **m** = number of classes  

- **Graph construction:** `O(n * m)`  
- **Flow algorithm (Ford-Fulkerson with shortest-path augmentations):**  
  - Each augmentation uses Dijkstra with a heap → `O(E log V)`  
  - Total = `O(F * E log V)`, where `F = n` (total flow of students).  

With `E ≈ n * m` and `V ≈ n + m`, complexity becomes:  
O(n^2 * m * log(n + m))

yaml
Copy
Edit

- **Space complexity:** `O(n * m)` to store the residual graph.  

➡️ Efficient for **moderate-scale university scheduling problems**.  

---

## 🛠️ Technologies Used
- **Python 3**  
- **Ford-Fulkerson framework** for maximum flow  
- **Heap-based Dijkstra** for augmenting shortest paths  
- **Min-Cost Max-Flow with lower bounds**  

---

## 📊 Applications
- University **course allocation** systems  
- **Scheduling classrooms** and time slots  
- **Resource distribution** under capacity constraints  

---

## 👨‍💻 Author
Developed as part of a **Data Structures & Algorithms assignment**, showcasing the application of
