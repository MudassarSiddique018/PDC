# Lab Guide — Parallel & Distributed Computing with MiniCluster ML

A step-by-step guide for running MiniCluster ML in a university lab with
three (or more) PCs, plus ready-made exercises that connect each lab-manual
topic — **Socket Programming → Multithreading → Multiprocessing / OpenMP →
MPI-style distributed computing** — to a real distributed application.

---

## Part A — Setting up the 3-PC lab

### A.1 Hardware & network

```
                 Teacher PC / any PC  (Coordinator)
                 IP e.g. 192.168.1.10
                 ports 8000 (web) + 9000 (workers)
                           │  lab switch / Wi-Fi
         ┌─────────────────┼─────────────────┐
       PC-01             PC-02             PC-03
     192.168.1.11      192.168.1.12      192.168.1.13
```

* All PCs on the same LAN (same switch or Wi-Fi network).
* The coordinator can also run on one of the three worker PCs.
* Find the coordinator's IP: `ipconfig` (Windows) or `ip addr` / `hostname -I` (Linux).

### A.2 Install (on every PC)

```bash
git clone https://github.com/<your-username>/minicluster-ml.git
cd minicluster-ml
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

> No internet in the lab? On a PC with internet run
> `pip download -r requirements.txt -d wheels`, copy the folder by USB and
> install with `pip install --no-index --find-links wheels -r requirements.txt`.

### A.3 Start the coordinator (one PC)

```bash
python -m minicluster.coordinator
# Windows double-click: scripts\start_coordinator.bat
```

Open **http://192.168.1.10:8000** in any browser in the lab.

**Windows firewall:** allow inbound TCP 8000 and 9000 (or click "Allow access"
when Windows asks the first time). Linux `ufw`: `sudo ufw allow 8000,9000/tcp`.

### A.4 Start a worker on every lab PC

```bash
python -m minicluster.worker --coordinator 192.168.1.10 --name PC-01
python -m minicluster.worker --coordinator 192.168.1.10 --name PC-02
python -m minicluster.worker --coordinator 192.168.1.10 --name PC-03
# Windows: scripts\start_worker.bat 192.168.1.10 PC-01
```

Within a second each PC appears on the dashboard as **ONLINE** with its live
CPU, RAM and per-core load.

Useful worker options:

| Option | Meaning |
|---|---|
| `--cores N` | how many CPU cores this worker may use (multiprocessing). Default: all. |
| `--secret S` | cluster secret, must match `MINICLUSTER_CLUSTER_SECRET` on the coordinator |
| `--slowdown 3` | pretend this PC is 3× slower (load-balancing experiments) |
| `--no-reconnect` | exit instead of reconnecting when the coordinator goes away |

### A.5 Only one PC available?

```bash
./scripts/run_local_cluster.sh            # Linux/macOS: coordinator + 3 workers
docker compose up --build                 # anywhere with Docker
```

Speed-up on one machine is limited by its **physical** cores — 3 workers on a
2-core laptop cannot go faster than ~2×. This is itself a nice discussion
point.

---

## Part B — Teaching scenario

> *"The university has 10 lakh (1,000,000) student records: matric marks,
> intermediate marks, entry test, attendance, previous GPA, weekly study
> hours and assignments submitted. It wants to predict whether a student will
> **pass or fail**. One computer takes too long. Can we make the 3 lab PCs
> work as a team?"*

The built-in **Student Performance** dataset implements exactly this
scenario and can generate any size from 1,000 to 5,000,000 rows. Three
targets are available:

| Target | Task | Model |
|---|---|---|
| `passed` | Fail / Pass | Logistic Regression |
| `category` | Fail / Pass / Distinction | Softmax Regression |
| `final_score` | 0–100 score | Linear Regression |

One-line summary for students:

> *"We have one big ML job. Instead of one computer doing it alone, we turn
> the three PCs into a team over the network. Each PC processes its share and
> the coordinator combines the results into the final answer."*

---

## Part C — Exercises

Each exercise lists the lab-manual concept, what to do in the dashboard, what
to observe, and questions for the lab report.

### Lab 1 — Socket programming: watch the protocol

**Concept:** client/server TCP sockets, message framing.

1. Read `minicluster/common/protocol.py` (≈150 lines).
2. Start the coordinator and **one** worker with `--log-level DEBUG`.
3. Stop and restart the coordinator — watch the worker reconnect with back-off.
4. Try `--secret wrong` on a worker when the coordinator has a secret set.

**Questions**
* Why does every frame start with two length fields? What would go wrong if
  we just called `recv(4096)` and parsed JSON?
* Why are NumPy arrays sent as a binary payload instead of JSON lists?

### Lab 2 — Multithreading inside a worker

**Concept:** threads sharing one socket, locks, producer/consumer queue.

1. Read `WorkerNode` in `minicluster/worker/node.py`: find the receiver,
   heartbeat and executor threads.
2. Submit a long prime-count job (`n = 5,000,000`). Notice the dashboard keeps
   updating CPU/RAM while the worker is busy — that is the heartbeat thread.

**Questions**
* What would happen without `_send_lock` when the heartbeat thread and the
  executor thread send at the same moment?
* Why is a `queue.Queue` used between the receiver and executor threads?

### Lab 3 — Level 1: distributed compute (scatter → compute → reduce)

**Concept:** data decomposition, reduction.

1. *New Job → Level 1*: **Square numbers**, `N = 10,000`, 3 workers.
2. Open the job: each PC got ranges of numbers; the coordinator reduced the
   partial sums. Check: Σi² = N(N+1)(2N+1)/6.
3. Repeat with **Count prime numbers**, `N = 2,000,000`.

**Questions**
* With 12 chunks and 3 workers, why does each worker *not* always get
  exactly 4 chunks?
* Prime counting near 2,000,000 is slower than near 1. How does
  `chunks_per_worker` help balance the load?

### Lab 4 — Multiprocessing / OpenMP-style parallelism inside one PC

**Concept:** shared-memory parallelism across CPU cores.

1. Restart one worker with `--cores 1`, then with `--cores 4`.
2. Run *Count primes* `N = 3,000,000, workers = 1` each time; compare durations.
3. On the job page look at the **Processes** column.

**Questions**
* The worker uses processes, not threads, for CPU work in Python. Why? (GIL)
* Training shards live in `shared_memory`. What would be copied per epoch
  without it?
* Map this design onto OpenMP: what corresponds to `#pragma omp parallel for`
  and to `reduction(+:sum)`?

### Lab 5 — Level 2: distributed machine learning

**Concept:** data parallelism, parameter server, synchronisation.

1. *New Job → Level 2*: dataset *Student Performance*, 300,000 rows, target
   `passed`, Logistic Regression, **Gradient sync**, 3 workers, 30 epochs.
2. Watch the per-worker progress bars and the loss curve.
3. Run the same job with **1 worker**. Compare the final weights in the
   *Result* panel — they are identical!
4. Now try **Model averaging** (lr 0.05, 10 epochs).
5. Use the trained model on the **Predict** page.

**Questions**
* Why is splitting data into 3 parts and training 3 *independent* models not
  the same as training one model on all the data?
* Gradient sync gives exactly the same model with 1 or 3 PCs. Prove it from
  the formula `g = Σ g_k / Σ n_k`.
* Model averaging sends fewer messages. What do we lose?

### Lab 6 — Level 3: speed-up, efficiency, Amdahl's law

**Concept:** performance analysis.

1. *New Job → Level 3*: **Count primes up to 2,000,000**.
2. Record the table: T(1), T(2), T(3), speed-up, efficiency.
3. Repeat with **Logistic regression, 20,000 students (communication bound)**.

| Workload | T(1) | T(2) | T(3) | S(3) | E(3) |
|---|---|---|---|---|---|
| primes 2M | | | | | |
| LR 20k rows | | | | | |
| LR 1M rows | | | | | |

**Questions**
* Why is the 3-PC time not exactly T(1)/3?
* Which workload scales better and why? Use the "Time breakdown"
  bar on the training job page.
* Estimate the serial fraction *f* from Amdahl's law
  `S(p) = 1 / (f + (1 − f)/p)` using your measurements.

### Lab 7 — Fault tolerance

**Concept:** failure detection, heartbeats, task reassignment.

1. Start a long training job (1,500,000 rows, 120 epochs, 3 workers).
2. While it runs, **unplug the network cable** of PC-02 (or press Ctrl+C in
   its terminal).
3. Watch the dashboard: `PC-02 OFFLINE` → `Task reassignment: PC-01 <- shard #1`.
4. The job still completes with the same accuracy. Plug PC-02 back in — it
   reconnects automatically.

**Questions**
* How long does the coordinator wait before declaring a PC dead after a cable
  is pulled? (Hint: `MINICLUSTER_HEARTBEAT_TIMEOUT`.) What is the trade-off
  of making it shorter?
* What extra work does the new owner of the lost shard have to do?

### Lab 8 — Load balancing

1. Start PC-03 with `--slowdown 3`.
2. Run *Count primes* `N = 3,000,000` with `chunks_per_worker = 8`.
3. Look at the *Chunks processed per worker* chart — the slow PC processes far
   fewer chunks. Now try `chunks_per_worker = 1`.

**Question:** what is the trade-off between many small chunks and few large
ones?

---

## Part D — Mapping to MPI

The coordinator/worker pattern here maps directly onto MPI collectives:

| MiniCluster ML | MPI equivalent |
|---|---|
| sending a data shard to each worker | `MPI_Scatter` |
| broadcasting weights every epoch | `MPI_Bcast` |
| summing gradients at the coordinator | `MPI_Reduce(..., MPI_SUM, root)` |
| sum + broadcast new weights | `MPI_Allreduce` |
| worker heartbeat / failure detection | *(not in classic MPI — one failure aborts the job)* |

A good extension project is to reimplement the training loop with `mpi4py`
(`comm.Allreduce(grad_local, grad_global, op=MPI.SUM)`) and compare it with
the socket version.
