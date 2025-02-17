# Global Distributed Hash Table for Word Counting using MPI and RDMA

A scalable and high-performance distributed hash table implementation designed for parallel word counting across multiple nodes. This project leverages **Message Passing Interface (MPI)** for efficient inter-node communication and **Remote Direct Memory Access (RDMA)** for low-latency data transfers, resulting in a 40% performance improvement.

## Features
- **Distributed Hash Table:** Efficient key-value storage and retrieval across multiple nodes.
- **Parallel Word Counting:** Processes large text datasets concurrently using distributed computing.
- **RDMA Integration:** Zero-copy data transfer mechanism to minimize communication overhead.
- **Scalable Design:** Handles increasing data volumes and node counts seamlessly.
- **Optimized Memory Access:** Improved memory utilization and reduced bottlenecks through direct memory operations.

## Architecture
- **MPI**: Handles message passing, synchronization, and task distribution across nodes.
- **RDMA**: Enables direct memory access between nodes without CPU intervention, reducing latency.
- **Hash Table Operations**: Distributed insertions, lookups, and updates with minimal contention.

## Performance Improvements
- Achieved a **40% increase in efficiency** through RDMA-based zero-copy data transfers.
- Reduced network latency and memory access overhead by optimizing communication patterns.

## Setup and Installation
1. **Prerequisites**:
   - MPI library (e.g., OpenMPI, MPICH)
   - RDMA-capable hardware and software stack (e.g., InfiniBand)
2. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/global-dht-wordcount.git
   cd global-dht-wordcount
   
## Build the project:
mpicc -o dht_wordcount dht_wordcount.c -lrdmacm -libverbs

## Run the application:
mpirun -np 4 ./dht_wordcount input.txt


